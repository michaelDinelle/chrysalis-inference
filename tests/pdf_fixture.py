"""PDFs for the document tests, written by hand (no PDF library): pages of white Helvetica text, which only the text layer
has, or of a picture filling the page, which only the drawn page has."""
import zlib


def pdf(pages):
    """A PDF of letter-sized pages: ("text", str) set in white Helvetica, or ("image", PIL image) as the whole page."""
    objs = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids, n = [], 4
    for kind, val in pages:
        if kind == "text":
            body, res = b"1 1 1 rg BT /F1 20 Tf 72 700 Td (" + val.encode("latin-1") + b") Tj ET", b"/Font << /F1 3 0 R >>"
        else:
            raw = zlib.compress(val.convert("RGB").tobytes())
            objs[n + 2] = (b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 "
                           b"/Filter /FlateDecode /Length %d >>\nstream\n" % (val.width, val.height, len(raw)) + raw + b"\nendstream")
            body, res = b"q 612 0 0 792 0 0 cm /Im1 Do Q", b"/XObject << /Im1 %d 0 R >>" % (n + 2)
        objs[n] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << %s >> /Contents %d 0 R >>" % (res, n + 1)
        objs[n + 1] = b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream"
        kids.append(n)
        n += 3
    objs[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    out, offs = bytearray(b"%PDF-1.4\n"), {}
    for i in sorted(objs):
        offs[i] = len(out)
        out += b"%d 0 obj\n" % i + objs[i] + b"\nendobj\n"
    xref, size = len(out), max(objs) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    out += b"".join(b"%010d 00000 n \n" % offs[i] if i in offs else b"0000000000 65535 f \n" for i in range(1, size))
    return bytes(out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref))
