"""Which pages of a PDF Chrysalis gives as pictures (chrysalis/documents.py), checked on PDFs whose pages say what they
are: each picture page a solid colour of its own, which the drawn page has to show. CPU only (poppler).
data/venv/bin/python tools/test_documents.py"""
import base64
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image

from chrysalis import apis, documents
from chrysalis.documents import DocumentError, Documents
from pdf_fixture import pdf

failed = 0


def check(name, ok, detail=""):
    global failed
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))


def colour(i):
    return ((37 * i) % 256, (91 * i + 40) % 256, (153 * i + 80) % 256)


def make(n, bare=()):
    """n pages: the ones in `bare` (1-based) a picture of colour(page), the others text naming the page."""
    return pdf([("image", Image.new("RGB", (64, 82), colour(p))) if p in bare else
                ("text", f"This is page {p} of the report, with enough words on it to count as text.") for p in range(1, n + 1)])


def pages_of(parts):
    """{page number: (its picture's mean colour or None, its text)} from a document's parts."""
    out, page, pic = {}, None, None
    for p in parts:
        if p["type"] == "text" and p["text"].startswith("<page number="):
            page, pic = int(p["text"].split('"')[1]), None
        elif p["type"] == "image_url":
            img = Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
            px = img.resize((1, 1), Image.BOX).getpixel((0, 0))
            pic = px
        elif page is not None and p["text"].endswith("</page>\n"):
            out[page] = (pic, p["text"])
    return out


def near(a, b):
    return a is not None and max(abs(x - y) for x, y in zip(a, b)) <= 3


def pictured(got):
    return sorted(p for p, (pic, _) in got.items() if pic is not None)


# 30 pages of text: the first 20 as pictures too, every page's text.
got = pages_of(documents.pdf_parts(make(30), "r.pdf", True))
check("a long PDF of text: its first 20 pages pictured", pictured(got) == list(range(1, 21)), str(pictured(got)))
check("... every page's text", all(f"page {p} of" in got[p][1] for p in range(1, 31)))

# Pages without text (scans) past the 20th are pictured wherever they are, and the budget goes to the pages with text.
bare = (5, 25, 28)
got = pages_of(documents.pdf_parts(make(30, bare), "r.pdf", True))
want = sorted(set(bare) | set([p for p in range(1, 31) if p not in bare][:20]))
check("pages without text pictured wherever they are", pictured(got) == want, f"{pictured(got)}, not {want}")
check("... each its own page", all(near(got[p][0], colour(p)) for p in bare), str({p: got[p][0] for p in bare}))

# page_pictures 0: text only, but for the pages that have nothing else.
got = pages_of(documents.pdf_parts(make(12, (3,)), "r.pdf", True, 0))
check("page_pictures 0: only the pages without text pictured", pictured(got) == [3], str(pictured(got)))

# A model without a vision tower gets the text alone.
got = pages_of(documents.pdf_parts(make(4, (2,)), "r.pdf", False))
check("no vision tower: no pictures", pictured(got) == [], str(pictured(got)))

# The cap on pages without text: those past it take the budget as the others do.
documents.SCANNED, kept = 4, documents.SCANNED
got = pages_of(documents.pdf_parts(make(9, tuple(range(1, 10))), "r.pdf", True, 2))
documents.SCANNED = kept
check("pages without text pictured up to SCANNED, then from the budget", pictured(got) == [1, 2, 3, 4, 5, 6], str(pictured(got)))

# page_pictures through each API's way of attaching a PDF.
doc = base64.b64encode(make(6)).decode()
url = "data:application/pdf;base64," + doc
chat = Documents().expand([{"role": "user", "content": [{"type": "file", "file": {"file_data": url, "filename": "r.pdf", "page_pictures": 2}}]}], True)
check("chat completions: page_pictures", pictured(pages_of(chat[0]["content"])) == [1, 2])
body = apis.messages_body({"model": "m", "max_tokens": 9, "messages": [{"role": "user", "content": [
    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": doc}, "page_pictures": 3}]}]})
msgs = Documents().expand(body["messages"], True)
check("Messages: page_pictures", pictured(pages_of([p for m in msgs for p in m["content"] if isinstance(m["content"], list)])) == [1, 2, 3])
body = apis.responses_body({"model": "m", "input": [{"role": "user", "content": [
    {"type": "input_file", "file_data": url, "filename": "r.pdf", "page_pictures": 1}]}]}, None)
msgs = Documents().expand(body["messages"], True)
check("Responses: page_pictures", pictured(pages_of([p for m in msgs for p in m["content"] if isinstance(m["content"], list)])) == [1])

for bad in (-1, 2.5, "all", True):
    try:
        Documents().expand([{"role": "user", "content": [{"type": "file", "file": {"file_data": url, "page_pictures": bad}}]}], True)
        check(f"page_pictures {bad!r} refused", False)
    except DocumentError:
        check(f"page_pictures {bad!r} refused", True)

print("all ok" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
