import base64
import hashlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image

from chrysalis import apis, documents
from chrysalis.documents import DocumentError, Documents
from pdf_fixture import parity, pdf

failed = 0


def check(name, ok, detail=""):
    global failed
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))


def colour(i):
    return ((37 * i) % 256, (91 * i + 40) % 256, (153 * i + 80) % 256)


def make(n, bare=()):
    return pdf([("image", Image.new("RGB", (64, 82), colour(p))) if p in bare else
                ("text", f"This is page {p} of the report, with enough words on it to count as text.") for p in range(1, n + 1)])


def pages_of(parts):
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


got = pages_of(documents.pdf_parts(make(30), "r.pdf", True))
check("a long PDF of text: its first 20 pages pictured", pictured(got) == list(range(1, 21)), str(pictured(got)))
check("... every page's text", all(f"page {p} of" in got[p][1] for p in range(1, 31)))

bare = (5, 25, 28)
got = pages_of(documents.pdf_parts(make(30, bare), "r.pdf", True))
want = sorted(set(bare) | set([p for p in range(1, 31) if p not in bare][:20]))
check("pages without text pictured wherever they are", pictured(got) == want, f"{pictured(got)}, not {want}")
check("... each its own page", all(near(got[p][0], colour(p)) for p in bare), str({p: got[p][0] for p in bare}))

got = pages_of(documents.pdf_parts(make(12, (3,)), "r.pdf", True, 0))
check("page_pictures 0: only the pages without text pictured", pictured(got) == [3], str(pictured(got)))

got = pages_of(documents.pdf_parts(make(4, (2,)), "r.pdf", False))
check("no vision tower: no pictures", pictured(got) == [], str(pictured(got)))

documents.SCANNED, kept = 4, documents.SCANNED
got = pages_of(documents.pdf_parts(make(9, tuple(range(1, 10))), "r.pdf", True, 2))
documents.SCANNED = kept
check("pages without text pictured up to SCANNED, then from the budget", pictured(got) == [1, 2, 3, 4, 5, 6], str(pictured(got)))

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


def blocks(img):
    a = np.asarray(img.convert("RGB"), np.float64)
    ys, xs = np.linspace(0, a.shape[0], 9).astype(int), np.linspace(0, a.shape[1], 9).astype(int)
    return np.array([[a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].reshape(-1, 3).mean(0) for j in range(8)] for i in range(8)])


def read(data, pictures):
    try:
        parts = documents.pdf_parts(data, "fixture.pdf", pictures)
    except DocumentError as e:
        return {"error": str(e)}
    pages = []
    for p in parts:
        if p["type"] == "text" and p["text"].startswith("<page number="):
            pages.append({"picture": None, "text": None})
        elif p["type"] == "image_url":
            pages[-1]["picture"] = Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].split(",", 1)[1])))
        elif pages and p["text"].endswith("</page>\n"):
            pages[-1]["text"] = p["text"][:-len("\n</page>\n")].strip()
    return {"pages": pages}


BLOCK_STEPS = 8
poppler = json.load(open(os.path.join(ROOT, "tests", "media", "poppler.json")))
for name, data in parity().items():
    check(f"{name}: the PDF poppler read", hashlib.sha256(data).hexdigest() == poppler[name]["sha256"])
    for mode in ("pictures", "no pictures"):
        if mode not in poppler[name]:
            continue
        was, got = poppler[name][mode], read(data, mode == "pictures")
        what = f"{name}, {mode}"
        if "error" in was:
            said = got.get("error", "")
            check(f"{what}: refused as poppler refused it", said.startswith("cannot read the PDF: ") if was["error"].startswith("cannot read the PDF: ")
                  else said == was["error"], f"{said!r}, poppler: {was['error'][:60]!r}")
            check(f"{what}: a password named only where poppler named one", ("password" in said) == ("password" in was["error"]), repr(said))
            continue
        pages = got.get("pages", [])
        shape = [(p["picture"]["size"] if p["picture"] else None) for p in was["pages"]]
        check(f"{what}: its pages, which are pictured and at what size", [list(p["picture"].size) if p["picture"] else None for p in pages] == shape,
              got.get("error", ""))
        same = [i + 1 for i, p in enumerate(pages) if i < len(was["pages"]) and p["text"].split() == was["pages"][i]["text"].split()]
        check(f"{what}: every page's words, in poppler's order", len(same) == len(was["pages"]) == len(pages),
              f"{len(same)} of {len(was['pages'])} pages")
        steps = [float(np.abs(blocks(p["picture"]) - np.array(was["pages"][i]["picture"]["thumb"])).max())
                 for i, p in enumerate(pages) if p["picture"] and i < len(was["pages"]) and was["pages"][i]["picture"]
                 and list(p["picture"].size) == was["pages"][i]["picture"]["size"]]
        check(f"{what}: each picture within {BLOCK_STEPS} steps of poppler's in every eighth", len(steps) == sum(s is not None for s in shape)
              and max(steps, default=0) <= BLOCK_STEPS, f"{len(steps)} pictures, largest {max(steps, default=0):.2f} steps")

print("all ok" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
