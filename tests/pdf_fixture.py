import hashlib
import struct
import zlib

PAD = bytes.fromhex("28bf4e5e4e758a4164004e56fffa01082e2e00b6d0683e802f0ca9fe6453697a")


def pdf(pages):
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


def rc4(key, data):
    s, j = list(range(256)), 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) % 256
        s[i], s[j] = s[j], s[i]
    out, i, j = bytearray(), 0, 0
    for b in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(b ^ s[(s[i] + s[j]) % 256])
    return bytes(out)


def numbers(values):
    return b" ".join(str(v).encode() for v in values)


def document(pages, inherited=None, owner=None, user=b"", count=None):
    objs, streams = {1: b"<< /Type /Catalog /Pages 2 0 R /AcroForm 3 0 R >>", 4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
                     5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"}, {}
    fields, kids, n = [], [], 6
    for page in pages:
        if page.get("missing"):
            kids.append(9999)
            continue
        me, n = n, n + 2
        streams[me + 1] = (b"", page.get("ops", b""))
        images = []
        for img in page.get("images", []):
            streams[n] = (b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8"
                          % (img.shape[1], img.shape[0]), img.tobytes())
            images.append(b"/Im%d %d 0 R" % (len(images) + 1, n))
            n += 1
        annots = []
        for rect, ops, widget in page.get("annots", []):
            streams[n + 1] = (b"/Type /XObject /Subtype /Form /BBox [0 0 %s] /Resources << /Font << /F1 4 0 R >> >>"
                              % numbers((rect[2] - rect[0], rect[3] - rect[1])), ops)
            kind = b"/Subtype /Widget /FT /Tx /T (field%d) /V (filled) /F 4" % n if widget else b"/Subtype /FreeText /DA (/F1 12 Tf) /F 4"
            objs[n] = b"<< /Type /Annot %s /Rect [%s] /P %d 0 R /AP << /N %d 0 R >> >>" % (kind, numbers(rect), me, n + 1)
            if widget:
                fields.append(b"%d 0 R" % n)
            annots.append(b"%d 0 R" % n)
            n += 2
        box = b"".join(b"/%s [%s] " % (name, numbers(page[key])) for key, name in (("media", b"MediaBox"), ("crop", b"CropBox")) if key in page)
        objs[me] = (b"<< /Type /Page /Parent 2 0 R %s%s/Resources << /Font << /F1 4 0 R /F2 5 0 R >> /XObject << %s >> >> /Contents %d 0 R%s >>"
                    % (box, b"/Rotate %d " % page["rotate"] if "rotate" in page else b"", b" ".join(images), me + 1,
                       b" /Annots [%s]" % b" ".join(annots) if annots else b""))
        kids.append(me)
    objs[2] = b"<< /Type /Pages %s/Kids [%s] /Count %d >>" % (b"/MediaBox [%s] " % numbers(inherited) if inherited else b"",
                                                            b" ".join(b"%d 0 R" % k for k in kids), count or len(kids))
    objs[3] = b"<< /Fields [%s] >>" % b" ".join(fields)
    trailer, key = b"", None
    if owner is not None:
        ident = hashlib.md5(b"chrysalis fixture").digest()
        o = rc4(hashlib.md5((owner + PAD)[:32]).digest()[:5], (user + PAD)[:32])
        key = hashlib.md5((user + PAD)[:32] + o + struct.pack("<i", -3904) + ident).digest()[:5]
        objs[n] = b"<< /Filter /Standard /V 1 /R 2 /O <%s> /U <%s> /P -3904 >>" % (o.hex().encode(), rc4(key, PAD).hex().encode())
        trailer = b" /Encrypt %d 0 R /ID [<%s> <%s>]" % (n, ident.hex().encode(), ident.hex().encode())
    for i, (head, body) in streams.items():
        if key is not None:
            body = rc4(hashlib.md5(key + i.to_bytes(3, "little") + b"\0\0").digest()[:10], body)
        objs[i] = b"<< %s /Length %d >>\nstream\n" % (head, len(body)) + body + b"\nendstream"
    out, offs = bytearray(b"%PDF-1.4\n"), {}
    for i in sorted(objs):
        offs[i] = len(out)
        out += b"%d 0 obj\n" % i + objs[i] + b"\nendobj\n"
    xref, size = len(out), max(objs) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    out += b"".join(b"%010d 00000 n \n" % offs[i] if i in offs else b"0000000000 65535 f \n" for i in range(1, size))
    return bytes(out + b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (size, trailer, xref))


def text(lines, size=11, x=72, y=720, lead=None):
    out = b"BT /F1 %g Tf %g TL %g %g Td" % (size, lead or size * 1.3, x, y)
    for line in lines:
        out += b" [%s] TJ T*" % line if isinstance(line, bytes) else b" (%s) Tj T*" % line.encode("latin-1")
    return out + b" ET"


def picture(w, h, seed):
    import numpy as np
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([(x * 255 // w + 40 * seed) % 256, (y * 255 // h + 70 * seed) % 256, ((x // 24 + y // 24) % 2) * 160 + 30 * seed % 90], -1)
    for k in range(5):
        cx, cy, r = (61 * k * seed) % w, (97 * k + 13 * seed) % h, min(w, h) // (4 + k)
        img[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = ((90 * k) % 256, (40 * k + 70 * seed) % 256, (150 + 20 * k) % 256)
    img[h // 2:h // 2 + h // 12, w // 10:w // 2] = 250
    return img.astype(np.uint8)


def spaced(words, x, y, size=11, gap=0.3):
    words, out, em = list(words), b"BT /F2 %g Tf" % size, 0.6 * size
    for i, word in enumerate(words):
        head, tail = (word[:-1], word[-1:]) if i + 1 < len(words) else (word, "")
        if head:
            out += b" 0 Tc 1 0 0 1 %g %g Tm (%s) Tj" % (x, y, head.encode("latin-1"))
            x += em * len(head)
        if tail:
            out += b" %g Tc 1 0 0 1 %g %g Tm (%s%s) Tj" % (gap * size, x, y, tail.encode("latin-1"), words[i + 1][:1].encode("latin-1"))
            x += 2 * em + gap * size
            words[i + 1] = words[i + 1][1:]
    return out + b" 0 Tc ET"


def frame(color, w, h):
    return b"q %s RG 6 w 10 10 %g %g re S %s rg 30 30 80 50 re f Q" % (color, w - 20, h - 20, color)


def parity():
    prose = ["Chrysalis reads every page of a document and pictures the ones without text.",
             "This line ends with a word broken across the line: inter-", "mediary results stay one word for the model.",
             b"(Words placed by) -320 (their gaps alone,) -300 (with no) -290 (space between them.)",
             "Numbers 1234.5678 and symbols like %, #, & and @ come through too.", "Short line."]
    docs = {}
    docs["report"] = document([
        {"ops": text(["Quarterly report"], size=20, y=740) + b" " + text(prose, y=700) + b" "
         + spaced(["Words", "kept", "apart", "by", "spacing", "alone."], 72, 560) + b" " + text(["Page 1"], size=9, x=520, y=40)},
        {"ops": b"q 0 0.4 0.8 rg 100 400 400 200 re f Q " + text(["Figure 2"], size=12, x=260, y=380)},
        {"ops": text(prose * 6, size=10, y=760, lead=12)},
        {"ops": text(["This line is far too long for its page, so its last words fall past the edge of the paper."], size=20, y=400)},
    ])
    docs["scans"] = document([
        {"ops": b"q 612 0 0 792 0 0 cm /Im1 Do Q", "images": [picture(306, 396, 1)]},
        {"ops": b"q 612 0 0 792 0 0 cm /Im1 Do Q BT 3 Tr /F1 14 Tf 72 600 Td (This scanned page carries a hidden layer of recognized text.) Tj ET",
         "images": [picture(306, 396, 2)]},
        {"ops": b"q 500 0 0 300 56 400 cm /Im1 Do Q " + text(["A caption"], x=56, y=380), "images": [picture(250, 150, 3)]},
        {"media": (0, 0, 595.276, 841.89), "ops": b"q 595.276 0 0 841.89 0 0 cm /Im1 Do Q", "images": [picture(298, 421, 4)]},
    ])
    docs["shapes"] = document([
        {"rotate": 90, "ops": frame(b"1 0 0", 612, 792) + b" " + text(["Turned a quarter, this page reads sideways on paper."])},
        {"rotate": 180, "ops": frame(b"0 0.6 0", 612, 792) + b" " + text(["Turned half way round, this page is upside down."])},
        {"rotate": 270, "ops": frame(b"0 0 1", 612, 792) + b" " + text(["Turned three quarters, this page reads the other way."])},
        {"crop": (50, 60, 400, 500), "ops": frame(b"0.5 0 0.5", 612, 792) + b" " + text(["Only part of this page is in its crop box."], x=60, y=480)},
        {"media": (-100, -50, 512, 742), "ops": frame(b"0 0.5 0.5", 412, 692) + b" " + text(["This page starts left of and below the origin."])},
        {"media": (0, 0, 1920, 1080), "ops": frame(b"0.8 0.4 0", 1920, 1080) + b" " + text(["A wide slide with a title on it"], size=48, x=200, y=900)},
        {"media": (0, 0, 1234.567, 89.01), "ops": b"0.2 0.5 0.7 rg -10 -10 1300 120 re f"},
        {"media": (0, 0, 100, 3000), "ops": b"0.9 0.1 0.4 rg -10 -10 120 3100 re f"},
    ])
    docs["turned"] = document([
        {"ops": text(["The heading of a page with a turned label beside its table."], y=740)
         + b" BT /F1 14 Tf 0 1 -1 0 300 200 Tm (Turned label text) Tj ET " + text(["Rows of the table go here."], y=500)},
    ])
    docs["annotated"] = document([
        {"ops": text(["The body of a page whose margin holds a note."], y=650),
         "annots": [((300, 600, 500, 630), b"q 1 1 0.6 rg 0 0 200 30 re f BT /F1 12 Tf 0 0 0 rg 6 10 Td (A note in the margin) Tj ET Q", False)]},
        {"ops": text(["Name of the entity filling in this form:"], y=700),
         "annots": [((72, 640, 400, 670), b"/Tx BMC q BT /F1 14 Tf 0 0 0 rg 4 8 Td (Anomaly Innovations Inc.) Tj ET Q EMC", True)]},
    ], inherited=(0, 0, 500, 700))
    docs["locked"] = document([{"ops": text(["Locked by an owner password, this text still reads without one."])},
                               {"ops": b"q 612 0 0 792 0 0 cm /Im1 Do Q", "images": [picture(204, 264, 5)]}], owner=b"owner")
    docs["password"] = document([{"ops": text(["Nobody reads this page without the password."])}], owner=b"owner", user=b"secret")
    docs["missing"] = document([{"ops": text(["The first page is whole, the second is missing from the file."])}, {"missing": True}])
    docs["counted"] = document([{"ops": text(["The page tree says there are four pages; there is one."])}], count=4)
    docs["empty"] = document([])
    docs["thousand"] = document([{"ops": text(["One of a thousand and one pages."])}] * 1001)
    docs["truncated"] = docs["report"][:len(docs["report"]) // 2]
    docs["not a pdf"] = b"%PDF-1.4\nthis file only starts like a PDF\n"
    return docs
