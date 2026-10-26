import struct, sys, collections

def read_gguf(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    s = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF"
    ver, nt, nkv = rd("I"), rd("Q"), rd("Q")
    scal = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def val(t):
        if t in scal: return rd(scal[t])
        if t == 8: return s()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [val(et) for _ in range(n)]
        raise ValueError(t)
    meta = {}
    for _ in range(nkv):
        k = s(); meta[k] = val(rd("I"))
    tensors = []
    for _ in range(nt):
        name = s(); nd = rd("I"); shape = [rd("Q") for _ in range(nd)]
        tensors.append((name, rd("I"), shape, rd("Q")))
    align = meta.get("general.alignment", 32)
    data_off = (f.tell() + align - 1) // align * align
    return meta, tensors, data_off

if __name__ == "__main__":
    meta, tensors, off = read_gguf(sys.argv[1])
    for k, v in meta.items():
        if isinstance(v, list): v = f"[{len(v)}] {v[:8]}"
        print(k, "=", v)
    print("data offset", off, "tensors", len(tensors))
    print("types:", collections.Counter(t[1] for t in tensors))
    pat = sys.argv[2] if len(sys.argv) > 2 else "blk.0."
    for t in tensors:
        if pat in t[0] or not t[0].startswith("blk."): print(*t)
