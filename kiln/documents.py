"""Documents attached to a message: a PDF reaches the model as Anthropic's API gives it one, each page both as the vision
tower sees it and as its text, here as text and picture parts of the message (pages past PAGES_SEEN as text alone); other
files as their text. poppler's pdftotext and pdftoppm read the PDF, in a scratch directory, with a time limit."""
import base64
import hashlib
import os
import re
import subprocess
import tempfile
import threading
from collections import OrderedDict

PAGES_SEEN = 20            # pages also given as pictures: ~800 tokens each at LONG_SIDE (a 10-page PDF ~10K with its text)
LONG_SIDE = 1024           # pixels along a page picture's longer side: 9-point text stays legible, and it has the text too
MAX_PAGES = 1000
TIMEOUT = 120              # seconds poppler may take over one PDF


class DocumentError(ValueError):
    """A document kiln cannot read."""


def file_bytes(data):
    """(bytes, media type) of a file part's data: a data URL or bare base64 (kiln has no file store to look ids up in)."""
    if not isinstance(data, str) or not data:
        raise DocumentError("files must be sent inline, as file_data (a data: URL or base64)")
    media = None
    if data.startswith("data:"):
        head, _, data = data.partition(",")
        media = head[5:].split(";")[0] or None
    try:
        return base64.b64decode(data, validate=False), media
    except ValueError as e:
        raise DocumentError(f"cannot decode the file data: {e}") from None


class Documents:
    """Expanded documents by content, so a conversation that sends the same PDF every turn renders it to the same parts
    (the same picture bytes, so the same image tokens and a prompt-cache hit) without reading it again."""

    def __init__(self, keep=16):
        self.keep, self.done, self.lock = keep, OrderedDict(), threading.Lock()

    def parts(self, data, media, name, pictures):
        """A file's content parts: text and image_url parts (pictures: the model has a vision tower)."""
        key = (hashlib.blake2b(data, digest_size=16).digest(), name, pictures)
        with self.lock:
            if key in self.done:
                self.done.move_to_end(key)
                return self.done[key]
        if data[:5] == b"%PDF-" or media == "application/pdf":
            out = pdf_parts(data, name, pictures)
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise DocumentError(f"kiln reads PDFs and text files; {name or 'this file'} is neither") from None
            out = [{"type": "text", "text": f'<document name="{name or "file"}">\n{text}\n</document>'}]
        with self.lock:
            self.done[key] = out
            while len(self.done) > self.keep:
                self.done.popitem(last=False)
        return out

    def expand(self, messages, pictures):
        """messages with each file part (OpenAI's chat completions: {"type": "file", "file": {"file_data", "filename"}})
        replaced by its document's parts."""
        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "file" for p in content):
                new = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "file":
                        f = p.get("file") or {}
                        if f.get("file_id"):
                            raise DocumentError("kiln has no file store: send files inline, as file_data")
                        new += self.parts(*file_bytes(f.get("file_data")), f.get("filename"), pictures)
                    else:
                        new.append(p)
                m = {**m, "content": new}
            out.append(m)
        return out


def pdf_parts(data, name, pictures):
    with tempfile.TemporaryDirectory(prefix="kiln-pdf-") as tmp:
        src = os.path.join(tmp, "in.pdf")
        with open(src, "wb") as f:
            f.write(data)
        info = poppler(["pdfinfo", src]).decode("utf-8", "replace")
        m = re.search(r"^Pages:\s+(\d+)", info, re.M)
        if not m:
            raise DocumentError("cannot read the PDF: it has no pages")
        pages = int(m.group(1))
        if pages > MAX_PAGES:
            raise DocumentError(f"the PDF has {pages} pages; kiln reads at most {MAX_PAGES}")
        texts = poppler(["pdftotext", "-enc", "UTF-8", src, "-"]).decode("utf-8", "replace").split("\f")
        seen = min(pages, PAGES_SEEN) if pictures else 0
        images = []
        if seen:
            poppler(["pdftoppm", "-png", "-scale-to", str(LONG_SIDE), "-l", str(seen), src, os.path.join(tmp, "page")])
            images = sorted(f for f in os.listdir(tmp) if f.startswith("page") and f.endswith(".png"))
            if len(images) != seen:
                raise DocumentError("cannot read the PDF: poppler drew fewer pages than it has")
        out = [{"type": "text", "text": f'<document name="{name or "document.pdf"}" pages="{pages}">\n'}]
        for i in range(pages):
            out.append({"type": "text", "text": f'<page number="{i + 1}">\n'})
            if i < seen:
                with open(os.path.join(tmp, images[i]), "rb") as f:
                    out.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(f.read()).decode()}})
            text = texts[i].strip() if i < len(texts) else ""
            out.append({"type": "text", "text": (f"\n{text}" if i < seen and text else text) + "\n</page>\n"})
        out.append({"type": "text", "text": "</document>"})
        return out


def poppler(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, check=True, timeout=TIMEOUT).stdout
    except subprocess.CalledProcessError as e:
        raise DocumentError(f"cannot read the PDF: {e.stderr.decode('utf-8', 'replace').strip()[:200]}") from None
    except subprocess.TimeoutExpired:
        raise DocumentError(f"reading the PDF took over {TIMEOUT} s") from None
    except FileNotFoundError:
        raise DocumentError(f"kiln needs poppler ({cmd[0]}) to read PDFs") from None
