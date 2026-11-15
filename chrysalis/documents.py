import base64
import hashlib
import io
import math
import threading
import time
from collections import OrderedDict

PAGES_SEEN = 20
SCANNED = 120
BARE = 32
LONG_SIDE = 1024
MAX_PAGES = 1000
TIMEOUT = 120
WORD_GAP, LINE_SHIFT = 0.2, 0.5
PDFIUM = threading.Lock()


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
    import pypdfium2 as pdfium
    import pypdfium2.raw as R
    with PDFIUM:
        R.FPDF_LoadMemDocument64(b"", 0, None)
        try:
            doc = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as e:
            reason = {R.FPDF_ERR_PASSWORD: "it needs a password", R.FPDF_ERR_SECURITY: "its encryption is not one Chrysalis reads"}
            raise DocumentError(f"cannot read the PDF: {reason.get(e.err_code, 'it is damaged or not a PDF')}") from None
        try:
            pages = len(doc)
            if pages > MAX_PAGES:
                raise DocumentError(f"the PDF has {pages} pages; Chrysalis reads at most {MAX_PAGES}")
            deadline = time.monotonic() + TIMEOUT
            texts, broken = [], set()
            for i in range(pages):
                in_time(deadline)
                page = prepared(pdfium, R, doc, i)
                if page is None:
                    broken.add(i)
                    texts.append("")
                else:
                    texts.append(page_text(R, page))
                    page.close()
            drawn = {}
            for i in picked(texts, pages, seen) if pictures else []:
                in_time(deadline)
                if i in broken:
                    raise DocumentError(f"cannot read the PDF: page {i + 1} is damaged")
                drawn[i] = drawing(pdfium, R, doc[i])
        finally:
            doc.close()
    out = [{"type": "text", "text": f'<document name="{name or "document.pdf"}" pages="{pages}">\n'}]
    for i in range(pages):
        out.append({"type": "text", "text": f'<page number="{i + 1}">\n'})
        if i in drawn:
            out.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(drawn[i]).decode()}})
        text = texts[i].strip()
        out.append({"type": "text", "text": (f"\n{text}" if i in drawn and text else text) + "\n</page>\n"})
    out.append({"type": "text", "text": "</document>"})
    return out


def in_time(deadline):
    if time.monotonic() > deadline:
        raise DocumentError(f"reading the PDF took over {TIMEOUT} s")


def prepared(pdfium, R, doc, i):
    try:
        page = doc[i]
    except pdfium.PdfiumError:
        return None
    page.set_cropbox(-1e9, -1e9, 1e9, 1e9)
    media = page.get_bbox()
    page.set_mediabox(*media)
    page.set_cropbox(*media)
    if R.FPDFPage_Flatten(page, R.FLAT_NORMALDISPLAY) == R.FLATTEN_SUCCESS:
        page.close()
        page = doc[i]
    return page


def page_text(R, page):
    left, bottom, right, top = page.get_bbox()
    tp = page.get_textpage()
    box, m = R.FS_RECTF(), R.FS_MATRIX()
    out, last, join, spaced = [], None, False, False
    n, j = R.FPDFText_CountChars(tp), -1
    while j + 1 < n:
        j += 1
        u = R.FPDFText_GetUnicode(tp, j)
        if u == 2:
            join = True
            continue
        if R.FPDFText_IsGenerated(tp, j):
            spaced = spaced or u == 32
            continue
        if 0xD800 <= u < 0xDC00 and j + 1 < n and 0xDC00 <= R.FPDFText_GetUnicode(tp, j + 1) < 0xE000:
            j += 1
            u = 0x10000 + ((u - 0xD800) << 10) + R.FPDFText_GetUnicode(tp, j) - 0xDC00
        c = chr(u)
        if c.isspace():
            spaced = True
            continue
        R.FPDFText_GetLooseCharBox(tp, j, box)
        if box.right < left or box.left > right or box.top < bottom or box.bottom > top:
            continue
        R.FPDFText_GetMatrix(tp, j, m)
        turn = round(math.atan2(m.b, m.a) / (math.pi / 2)) % 4
        start, end, base = ((box.left, box.right, m.f), (box.bottom, box.top, m.e),
                            (-box.right, -box.left, m.f), (-box.top, -box.bottom, m.e))[turn]
        em = R.FPDFText_GetFontSize(tp, j) * math.hypot(m.c, m.d)
        if last is not None and not join:
            unit = max(em, last[3])
            if turn != last[0] or abs(base - last[2]) >= LINE_SHIFT * unit or start < last[1] - unit:
                out.append("\n")
            elif spaced or start - last[1] > WORD_GAP * unit:
                out.append(" ")
        join = spaced = False
        out.append(c)
        last = (turn, end, base, em)
    tp.close()
    return "".join(out)


def drawing(pdfium, R, page):
    import numpy as np
    left, bottom, right, top = (float(str(np.float32(v))) for v in page.get_bbox())
    w, h = right - left, top - bottom
    if page.get_rotation() in (90, 270):
        w, h = h, w
    scale = (72.0 * LONG_SIDE) / (w if w > h else h) / 72.0
    width, height = math.ceil(w * scale), math.ceil(h * scale)
    bitmap = pdfium.PdfBitmap.new_native(width, height, R.FPDFBitmap_BGR, rev_byteorder=True)
    try:
        bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
        R.FPDF_RenderPageBitmapWithMatrix(bitmap, page, R.FS_MATRIX(scale, 0, 0, scale, 0, 0), R.FS_RECTF(0, 0, width, height),
                                          R.FPDF_ANNOT | R.FPDF_REVERSE_BYTE_ORDER)
        png = io.BytesIO()
        bitmap.to_pil().save(png, "PNG", compress_level=1)
    finally:
        bitmap.close()
    page.close()
    return png.getvalue()
