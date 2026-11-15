import io


def film():
    import av
    from PIL import Image, ImageDraw, ImageFont
    out, font = io.BytesIO(), ImageFont.load_default(size=48)
    with av.open(out, "w", format="mp4") as box:
        stream = box.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for i in range(40):
            img = Image.new("RGB", (640, 360), "white")
            d = ImageDraw.Draw(img)
            if i < 20:
                d.ellipse([200 + 4 * i, 100, 320 + 4 * i, 220], fill="red")
            else:
                d.rectangle([100, 120, 220, 240], fill="blue")
                d.text((300, 150), "CHRYSALIS 58", fill="black", font=font)
            box.mux(stream.encode(av.VideoFrame.from_image(img)))
        box.mux(stream.encode())
    return out.getvalue()
