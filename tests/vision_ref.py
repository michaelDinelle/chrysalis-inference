import json, os, sys, types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cmd, src, dst = sys.argv[1:4]
if cmd == "processor":
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    cfg = json.load(open(f"{ROOT}/data/models/vision/preprocessor_config.json"))
    extra = {"max_pixels": int(sys.argv[4])} if len(sys.argv) > 4 else {}
    proc = Qwen2VLImageProcessor(**{k: v for k, v in cfg.items() if k not in ("processor_class", "image_processor_type")}, **extra)
    out = proc(images=[Image.open(src).convert("RGB")], return_tensors="np")
    pv = out["pixel_values"].reshape(-1, 3, 2, 16, 16)
    assert np.array_equal(pv[:, :, 0], pv[:, :, 1])
    np.savez(dst, patches=pv[:, :, 0].reshape(-1, 768), grid=out["image_grid_thw"][0][1:])
elif cmd == "tower":
    import torch
    from safetensors import safe_open
    torch.set_num_threads(16)
    if sys.argv[4:5] == ["fn"]:
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig as Config
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel as Tower
        config, shard = "data/models/vision/flash-next-config.json", "data/models/vision/qwen3.8-flash-next-shard1.safetensors"
    else:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config as Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel as Tower
        config, shard = "data/models/config.json", "data/models/vision/qwen3.8-27b-shard1.safetensors"
    vcfg = Config.from_dict(json.load(open(f"{ROOT}/{config}"))).vision_config
    vcfg._attn_implementation = "sdpa"
    model = Tower(vcfg).float().eval()
    with safe_open(f"{ROOT}/{shard}", "pt") as f:
        missing, unexpected = model.load_state_dict({k[len("model.visual."):]: f.get_tensor(k).float() for k in f.keys()
                                                     if k.startswith("model.visual.")}, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    z = np.load(src)
    if z["patches"].shape[1] == 1536:
        pv, grid = torch.from_numpy(z["patches"]).contiguous(), torch.tensor([[int(g) for g in z["grid"]]])
    else:
        pv = torch.from_numpy(z["patches"]).reshape(-1, 3, 1, 16, 16).expand(-1, -1, 2, -1, -1).reshape(-1, 1536).contiguous()
        grid = torch.tensor([[1, int(z["grid"][0]), int(z["grid"][1])]])
    with torch.no_grad():
        out = model(pv, grid_thw=grid).pooler_output
    np.save(dst, out.float().numpy())
elif cmd == "video":
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    from transformers.video_utils import VideoMetadata
    z = np.load(src)
    frames = z["frames"]
    proc = Qwen3VLVideoProcessor(size={"shortest_edge": 4096, "longest_edge": 24576 * 32 * 32}, fps=2, min_frames=4, max_frames=768,
                                 cap_pixels_per_frame=True, max_video_tokens=768, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    meta = VideoMetadata(total_num_frames=len(frames), fps=float(z["fps"]), width=frames.shape[2], height=frames.shape[1])
    out = proc(videos=[frames], video_metadata=[meta], do_sample_frames=True, return_metadata=True, return_tensors="np")
    m = out["video_metadata"][0]
    stub = types.SimpleNamespace(video_processor=proc, vision_start_token="<|vision_start|>", vision_end_token="<|vision_end|>",
                                 video_token="<|video_pad|>")
    stub._calculate_timestamps = lambda *a: Qwen3VLProcessor._calculate_timestamps(stub, *a)
    text = Qwen3VLProcessor.replace_video_token(stub, {"video_grid_thw": out["video_grid_thw"], "video_metadata": [m]}, 0)
    np.savez(dst, indices=np.array(m.frames_indices), patches=out["pixel_values_videos"], grid=out["video_grid_thw"][0], text=np.array(text))
elif cmd == "rope":
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    z = np.load(src)
    stub = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    stub.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(stub, *a, **k)
    kinds = torch.from_numpy(z["types"]).long()[None]
    pos, _ = Qwen3_5Model.get_rope_index(stub, torch.zeros_like(kinds), kinds, video_grid_thw=torch.from_numpy(z["grids"]).long())
    np.save(dst, pos[:, 0].T.numpy().astype(np.int32))
