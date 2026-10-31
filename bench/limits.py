import ctypes as C
import glob
import json
import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRANSFERS = {9: ("GDDR6", 16), 10: ("DDR5", 2)}
MATRIX_FP16 = {9: {0x908: 1024, 0x90a: 1024, 0x940: 2048, 0x941: 2048, 0x942: 2048, 0x950: 2048}, 10: 256, 11: 512, 12: 1024}


class GpuInfo(C.Structure):
    _fields_ = [("asic_id", C.c_uint32), ("chip_rev", C.c_uint32), ("chip_external_rev", C.c_uint32), ("family_id", C.c_uint32),
                ("ids_flags", C.c_uint64), ("max_engine_clk", C.c_uint64), ("max_memory_clk", C.c_uint64),
                ("num_shader_engines", C.c_uint32), ("num_shader_arrays_per_engine", C.c_uint32), ("avail_quad_shader_pipes", C.c_uint32),
                ("max_quad_shader_pipes", C.c_uint32), ("cache_entries_per_quad_pipe", C.c_uint32), ("num_hw_gfx_contexts", C.c_uint32),
                ("rb_pipes", C.c_uint32), ("enabled_rb_pipes_mask", C.c_uint32), ("gpu_counter_freq", C.c_uint32),
                ("backend_disable", C.c_uint32 * 4), ("mc_arb_ramcfg", C.c_uint32), ("gb_addr_cfg", C.c_uint32),
                ("gb_tile_mode", C.c_uint32 * 32), ("gb_macro_tile_mode", C.c_uint32 * 16), ("pa_sc_raster_cfg", C.c_uint32 * 4),
                ("pa_sc_raster_cfg1", C.c_uint32 * 4), ("cu_active_number", C.c_uint32), ("cu_ao_mask", C.c_uint32),
                ("cu_bitmap", (C.c_uint32 * 4) * 4), ("vram_type", C.c_uint32), ("vram_bit_width", C.c_uint32), ("ce_ram_size", C.c_uint32),
                ("vce_harvest_config", C.c_uint32), ("pci_rev_id", C.c_uint32)]


def props(path):
    return {k: int(v) for k, v in (line.split() for line in open(path) if len(line.split()) == 2)}


def pci_path(node):
    p = props(f"{node}/properties")
    loc = p["location_id"]
    return f"/sys/bus/pci/devices/{p.get('domain', 0):04x}:{loc >> 8:02x}:{(loc >> 3) & 31:02x}.{loc & 7}"


def amdgpu(pci):
    drm = C.CDLL("libdrm_amdgpu.so.1")
    drm.amdgpu_get_marketing_name.restype = C.c_char_p
    node = next(f"/dev/dri/{os.path.basename(r)}" for r in glob.glob(f"{pci}/drm/renderD*"))
    fd = os.open(node, os.O_RDWR)
    try:
        major, minor, dev = C.c_uint32(), C.c_uint32(), C.c_void_p()
        if drm.amdgpu_device_initialize(fd, C.byref(major), C.byref(minor), C.byref(dev)):
            raise OSError(f"amdgpu_device_initialize failed on {node}")
        info = GpuInfo()
        drm.amdgpu_query_gpu_info(dev, C.byref(info))
        name = (drm.amdgpu_get_marketing_name(dev) or b"").decode()
        drm.amdgpu_device_deinitialize(dev)
        return info, name
    finally:
        os.close(fd)


def memory(info):
    kind, per_clock = TRANSFERS.get(info.vram_type, (f"memory type {info.vram_type}", None))
    bw = info.max_memory_clk * 1e3 * per_clock * info.vram_bit_width / 8 if per_clock else None
    return kind, bw


def pcie(pci):
    gts = float(open(f"{pci}/current_link_speed").read().split()[0])
    lanes = int(open(f"{pci}/current_link_width").read())
    return gts * 1e9 * lanes * (128 / 130 if gts >= 8 else 0.8) / 8, f"PCIe {gts:g} GT/s x{lanes}"


NVIDIA_FP16 = {(7, 0): 1024, (7, 2): 1024, (7, 5): 1024, (8, 0): 2048, (8, 6): 1024, (8, 7): 1024, (8, 9): 1024, (9, 0): 4096,
               (10, 0): 8192, (10, 3): 8192, (12, 0): 1024, (12, 1): 1024}
PCIE_GTS = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0}


def nvidia():
    try:
        nv, cu = C.CDLL("libnvidia-ml.so.1"), C.CDLL("libcuda.so.1")
    except OSError:
        return None
    if nv.nvmlInit_v2() or cu.cuInit(0):
        return None
    count = C.c_uint(0)
    nv.nvmlDeviceGetCount_v2(C.byref(count))
    gpus = []
    for i in range(count.value):
        h, dev = C.c_void_p(), C.c_int()
        if nv.nvmlDeviceGetHandleByIndex_v2(i, C.byref(h)) or cu.cuDeviceGet(C.byref(dev), i):
            continue
        name = C.create_string_buffer(96)
        nv.nvmlDeviceGetName(h, name, 96)
        attr = lambda a: (lambda v: (cu.cuDeviceGetAttribute(C.byref(v), a, dev), v.value)[1])(C.c_int())
        sms, major, minor = attr(16), attr(75), attr(76)
        clock = lambda k: (lambda v: (nv.nvmlDeviceGetMaxClockInfo(h, k, C.byref(v)), v.value)[1])(C.c_uint())
        sm_mhz, mem_mhz = clock(1), clock(2)
        bits, gen, width = C.c_uint(), C.c_uint(), C.c_uint()
        nv.nvmlDeviceGetMemoryBusWidth(h, C.byref(bits))
        nv.nvmlDeviceGetMaxPcieLinkGeneration(h, C.byref(gen))
        nv.nvmlDeviceGetMaxPcieLinkWidth(h, C.byref(width))
        per_sm = NVIDIA_FP16.get((major, minor))
        gts = PCIE_GTS.get(gen.value)
        gpus.append({"name": name.value.decode(), "gfx": f"sm_{major}{minor}", "cus": sms, "clock_mhz": sm_mhz, "memory": "VRAM",
                     "bus_bits": bits.value, "memory_clock_mhz": mem_mhz, "bandwidth": mem_mhz * 1e6 * 2 * bits.value / 8,
                     "fp16": sms * sm_mhz * 1e6 * per_sm if per_sm else None, "integrated": False,
                     "pcie": gts * 1e9 * width.value * (128 / 130 if gts >= 8 else 0.8) / 8 if gts else None,
                     "link": f"PCIe {gts:g} GT/s x{width.value}" if gts else "PCIe"})
    return {"gpus": gpus, "ram": None} if gpus else None


APPLE_BANDWIDTH = {("Apple M1", None): 68.25e9, ("Apple M1 Pro", None): 200e9, ("Apple M1 Max", None): 400e9,
                   ("Apple M1 Ultra", None): 800e9, ("Apple M2", None): 100e9, ("Apple M2 Pro", None): 200e9, ("Apple M2 Max", None): 400e9,
                   ("Apple M2 Ultra", None): 800e9, ("Apple M3", None): 100e9, ("Apple M3 Pro", None): 150e9, ("Apple M3 Max", 30): 300e9,
                   ("Apple M3 Max", 40): 400e9, ("Apple M3 Ultra", None): 819e9, ("Apple M4", None): 120e9, ("Apple M4 Pro", None): 273e9,
                   ("Apple M4 Max", 32): 410e9, ("Apple M4 Max", 40): 546e9, ("Apple M5", None): 153e9, ("Apple M5 Pro", None): 307e9,
                   ("Apple M5 Max", 32): 460e9, ("Apple M5 Max", 40): 614e9, ("Apple M5 Ultra", None): 1200e9}


def apple():
    read = lambda *cmd: subprocess.run(cmd, capture_output=True, text=True).stdout
    acc = read("ioreg", "-r", "-d", "1", "-c", "IOAccelerator")
    name = re.search(r'"model" = "([^"]+)"', acc).group(1)
    cores = int(re.search(r'"gpu-core-count" = (\d+)', acc).group(1))
    states = re.search(r'"voltage-states9" = <([0-9a-f]+)>', read("ioreg", "-l", "-w0", "-p", "IODeviceTree", "-n", "pmgr"))
    raw = bytes.fromhex(states.group(1)) if states else b""
    clock = max((struct.unpack_from("<I", raw, i)[0] for i in range(0, len(raw) - 7, 8)), default=0)
    bw = APPLE_BANDWIDTH.get((name, cores), APPLE_BANDWIDTH.get((name, None)))
    per_lane = 4 if name.startswith("Apple M4") else 2
    gpu = {"name": name, "gfx": "apple", "cus": cores, "clock_mhz": clock / 1e6, "memory": "unified", "bus_bits": None,
           "memory_clock_mhz": None, "bandwidth": bw, "fp16": cores * 128 * per_lane * clock if clock else None, "integrated": True,
           "pcie": None, "link": "unified memory"}
    return {"gpus": [gpu], "ram": {"source": name, "memory": "unified", "bus_bits": None, "memory_clock_mhz": None, "bandwidth": bw}}


def machine():
    if sys.platform == "darwin":
        return apple()
    if os.path.exists("/dev/nvidiactl"):
        found = nvidia()
        if found:
            return found
    gpus, cpu_ram = [], None
    for node in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*")):
        p = props(f"{node}/properties")
        if not p.get("simd_count"):
            continue
        pci = pci_path(node)
        info, name = amdgpu(pci)
        version = p["gfx_target_version"]
        major, arch = version // 10000, (version // 10000) * 0x100 + ((version // 100) % 100) * 0x10 + version % 100
        per_cu = MATRIX_FP16.get(major)
        per_cu = per_cu.get(arch) if isinstance(per_cu, dict) else per_cu
        cus = p["simd_count"] // p["simd_per_cu"]
        kind, bw = memory(info)
        entry = {"name": name, "gfx": f"gfx{major}{(version // 100) % 100:x}{version % 100:x}", "cus": cus,
                 "clock_mhz": p["max_engine_clk_fcompute"], "memory": kind, "bus_bits": info.vram_bit_width,
                 "memory_clock_mhz": info.max_memory_clk / 1e3, "bandwidth": bw,
                 "fp16": cus * p["max_engine_clk_fcompute"] * 1e6 * per_cu if per_cu else None, "integrated": bool(p.get("cpu_cores_count"))}
        if info.vram_type in (10, 11, 12):
            cpu_ram = {"source": name, "memory": kind, "bus_bits": info.vram_bit_width, "memory_clock_mhz": info.max_memory_clk / 1e3,
                       "bandwidth": bw}
        else:
            entry["pcie"], entry["link"] = pcie(pci)
            gpus.append(entry)
    return {"gpus": gpus, "ram": cpu_ram}


def field(r, key):
    f = r.fields[key]
    return f.parts[f.data[0]].tolist()[0]


def dense(path, kv_bytes):
    from gguf import GGUFReader
    r = GGUFReader(path)
    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    k = lambda name: field(r, f"{arch}.{name}")
    layers, every = k("block_count") - k("nextn_predict_layers"), k("full_attention_interval")
    heads, head_dim = k("attention.head_count"), k("attention.key_length")
    v_heads, state = k("ssm.time_step_rank"), k("ssm.state_size")
    main = [t for t in r.tensors if (m := re.match(r"blk\.(\d+)\.", t.name)) and int(m.group(1)) < layers]
    head = next(t for t in r.tensors if t.name == "output.weight")
    attn = sum(1 for l in range(layers) if l % every == every - 1)
    return {"params": sum(int(t.n_elements) for t in main if len(t.shape) == 2), "head_params": int(head.n_elements),
            "weights": sum(int(t.n_bytes) for t in main) + int(head.n_bytes) + sum(int(t.n_bytes) for t in r.tensors if t.name == "output_norm.weight"),
            "attn_layers": attn, "heads": heads, "head_dim": head_dim, "recurrent_flops": 7 * v_heads * state * state * (layers - attn),
            "state": 2 * 4 * v_heads * state * state * (layers - attn), "kv": kv_bytes * attn}


def moe(path, kv_bytes, experts=512, used=10, layers=48, every=4, heads=24, head_dim=256, v_heads=48, state=128, selected=2048,
        pool=4, index_heads=4, index_dim=128):
    from gguf import GGUFReader
    r = GGUFReader(path)
    main = [t for t in r.tensors if (m := re.match(r"blk\.(\d+)\.", t.name)) and int(m.group(1)) < layers]
    routed = [t for t in main if re.search(r"\.ffn_(gate|up|down)_(hot|cold\d*|exps)\.weight$", t.name)]
    rest = [t for t in main if t not in routed]
    top = [t for t in r.tensors if not t.name.startswith(("blk.", "token_embd", "per_layer", "ple"))]
    share = used / experts
    attn = layers // every
    return {"params": sum(int(t.n_elements) for t in rest if len(t.shape) >= 2) + share * sum(int(t.n_elements) for t in routed),
            "head_params": int(next(t for t in r.tensors if t.name == "output.weight").n_elements),
            "weights": sum(int(t.n_bytes) for t in rest + top) + share * sum(int(t.n_bytes) for t in routed),
            "attn_layers": attn, "heads": heads, "head_dim": head_dim, "selected": selected, "pool": pool, "index_heads": index_heads,
            "index_dim": index_dim,
            "recurrent_flops": 7 * v_heads * state * state * (layers - attn), "state": 2 * 4 * v_heads * state * state * (layers - attn),
            "kv": kv_bytes * attn}


def attended(c, pos):
    return min(pos, c["selected"]) if "selected" in c else pos


def prompt_limit(c, g, start, n):
    flops = 2 * c["params"] * n + c["recurrent_flops"] * n + 2 * c["head_params"]
    if "selected" in c:
        sel = c["selected"]
        full = sum(min(p, sel) for p in range(start, start + n)) if n < 4096 else \
            (max(0, min(start + n, sel) - start) * (start + min(start + n, sel)) / 2 + max(0, start + n - max(start, sel)) * sel)
        flops += 4 * c["heads"] * c["head_dim"] * c["attn_layers"] * full
        flops += 2 * c["index_heads"] * c["index_dim"] * c["attn_layers"] * (n * start + n * (n - 1) / 2) / c["pool"]
    else:
        flops += 4 * c["heads"] * c["head_dim"] * c["attn_layers"] * (n * start + n * (n - 1) / 2)
    return max(flops / g["fp16"], (c["weights"] + c["state"] + c["kv"] * attended(c, start + n)) / g["bandwidth"])


def write_limit(c, g, context, per_pass):
    index = 2 * c["index_dim"] * c["attn_layers"] * context / c["pool"] if "selected" in c else 0
    return g["bandwidth"] / (c["weights"] + c["state"] + c["kv"] * attended(c, context) + index) * per_pass


CASE_IDS = {"short": "performance-short-r1", "4k": "performance-prompt-4k-r1", "16k": "performance-prompt-16k-r1", "100k": "long-100k",
            "followup": "follow-up"}
KV = {"hip": {"27b": 1792, "fn": 896}, "cuda": {"27b": 1792, "fn": 896}, "metal": {"27b": 2176, "fn": 2112}}


def costs(name):
    from measure import BENCH
    path = os.path.join(ROOT, BENCH[name]["files"][0])
    from chrysalis.models import BACKEND
    kv = KV[BACKEND][name]
    return dense(path, kv) if name == "27b" else moe(path, kv)


def shares(name, passes, g=None, c=None):
    import statistics
    g, c = g or machine()["gpus"][0], c or costs(name)
    out = {}
    for case, rid in CASE_IDS.items():
        rows = [(a, p["metrics"]) for p in passes for a in p.get("answers", []) if a["id"] == rid]
        if not rows:
            continue
        a, m = rows[0]
        new = a["prompt"] - a["cached"]
        prompt = statistics.fmean(x[f"prompt_tps.{case}"] for _, x in rows) / (new / prompt_limit(c, g, a["cached"], new))
        write = statistics.fmean(x["decode_tps"] / write_limit(c, g, x["prompt"] + x["written"] / 2, x["per_pass"]) for x, _ in rows)
        out[case] = (prompt, write)
    return out


def show(m):
    for g in m["gpus"]:
        print(f"{g['name']} ({g['gfx']}): {g['cus']} {'GPU cores' if g['gfx'] == 'apple' else 'SMs' if g['gfx'].startswith('sm_') else 'CUs'} at {g['clock_mhz']:.0f} MHz, "
              + (f"{g['memory']} {g['bus_bits']}-bit at {g['memory_clock_mhz']:.0f} MHz" if g["bus_bits"] else f"{g['memory']} memory"))
        print(f"  memory bandwidth {g['bandwidth'] / 1e9:.1f} GB/s, FP16 matrix {g['fp16'] / 1e12:.1f} TFLOPS, {g['link']}"
              + (f" {g['pcie'] / 1e9:.1f} GB/s each way" if g["pcie"] else ""))
    r = m["ram"]
    if r and not r["bus_bits"]:
        print(f"System RAM: the GPU's own, {r['bandwidth'] / 1e9:.1f} GB/s")
    elif r:
        print(f"System RAM ({r['memory']} {r['bus_bits']}-bit at {r['memory_clock_mhz']:.0f} MHz, read from {r['source']}): "
              f"{r['bandwidth'] / 1e9:.1f} GB/s")
    else:
        print("System RAM: no memory controller on this machine reports its speed without root")


if __name__ == "__main__":
    m = machine()
    print(json.dumps(m, indent=1) if "--json" in sys.argv else "", end="")
    if "--json" not in sys.argv:
        show(m)
