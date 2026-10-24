import base64
import hashlib
import os
import re
import subprocess
import tempfile
import threading
from collections import OrderedDict

PAGES_SEEN = 20
SCANNED = 120
BARE = 32
LONG_SIDE = 1024
MAX_PAGES = 1000
TIMEOUT = 120


class DocumentError(ValueError):
    pass


def file_bytes(data):
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

    def __init__(self, keep=16):
        self.keep, self.done, self.lock = keep, OrderedDict(), threading.Lock()

    def parts(self, data, media, name, pictures, seen=None):
        key = (hashlib.blake2b(data, digest_size=16).digest(), name, pictures, seen)
        with self.lock:
            if key in self.done:
                self.done.move_to_end(key)
                return self.done[key]
        if data[:5] == b"%PDF-" or media == "application/pdf":
            out = pdf_parts(data, name, pictures, PAGES_SEEN if seen is None else seen)
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise DocumentError(f"Chrysalis reads PDFs and text files; {name or 'this file'} is neither") from None
            out = [{"type": "text", "text": f'<document name="{name or "file"}">\n{text}\n</document>'}]
        with self.lock:
            self.done[key] = out
            while len(self.done) > self.keep:
                self.done.popitem(last=False)
        return out

    def expand(self, messages, pictures):
        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "file" for p in content):
                new = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "file":
                        f = p.get("file") or {}
                        if f.get("file_id"):
                            raise DocumentError("Chrysalis has no file store: send files inline, as file_data")
                        new += self.parts(*file_bytes(f.get("file_data")), f.get("filename"), pictures, page_pictures(f))
                    else:
                        new.append(p)
                m = {**m, "content": new}
            out.append(m)
        return out


def page_pictures(f):
    n = f.get("page_pictures")
    if n is None:
        return None
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise DocumentError("page_pictures must be a whole number of pages, 0 or more")
    return min(n, MAX_PAGES)


def picked(texts, pages, seen):
    bare = [i for i in range(pages) if len("".join((texts[i] if i < len(texts) else "").split())) < BARE][:SCANNED]
    rest = [i for i in range(pages) if i not in set(bare)][:seen]
    return sorted(bare + rest)


def pdf_parts(data, name, pictures, seen=PAGES_SEEN):
    with tempfile.TemporaryDirectory(prefix="chrysalis-pdf-") as tmp:
        src = os.path.join(tmp, "in.pdf")
        with open(src, "wb") as f:
            f.write(data)
        info = poppler(["pdfinfo", src]).decode("utf-8", "replace")
        m = re.search(r"^Pages:\s+(\d+)", info, re.M)
        if not m:
            raise DocumentError("cannot read the PDF: it has no pages")
        pages = int(m.group(1))
        if pages > MAX_PAGES:
            raise DocumentError(f"the PDF has {pages} pages; Chrysalis reads at most {MAX_PAGES}")
        texts = poppler(["pdftotext", "-enc", "UTF-8", src, "-"]).decode("utf-8", "replace").split("\f")
        drawn = {}
        if pictures:
            want = picked(texts, pages, seen)
            runs = []
            for i in want:
                if runs and runs[-1][1] == i - 1:
                    runs[-1][1] = i
                else:
                    runs.append([i, i])
            for a, b in runs:
                poppler(["pdftoppm", "-png", "-scale-to", str(LONG_SIDE), "-f", str(a + 1), "-l", str(b + 1), src,
                         os.path.join(tmp, f"run{a}")])
            for f in os.listdir(tmp):
                m = re.fullmatch(r"run\d+-0*(\d+)\.png", f)
                if m:
                    drawn[int(m.group(1)) - 1] = os.path.join(tmp, f)
            if sorted(drawn) != want:
                raise DocumentError("cannot read the PDF: poppler drew fewer pages than it has")
        out = [{"type": "text", "text": f'<document name="{name or "document.pdf"}" pages="{pages}">\n'}]
        for i in range(pages):
            out.append({"type": "text", "text": f'<page number="{i + 1}">\n'})
            if i in drawn:
                with open(drawn[i], "rb") as f:
                    out.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(f.read()).decode()}})
            text = texts[i].strip() if i < len(texts) else ""
            out.append({"type": "text", "text": (f"\n{text}" if i in drawn and text else text) + "\n</page>\n"})
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
        raise DocumentError(f"Chrysalis needs poppler ({cmd[0]}) to read PDFs") from None
