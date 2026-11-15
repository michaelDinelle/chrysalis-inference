# What the releases carry

Chrysalis's own code is under the MIT license (LICENSE). Its releases carry other people's work, each under its own
license, with the license texts in the package folders they come in:

- A Python (PSF license) with numpy, tokenizers, Pillow, llguidance, huggingface_hub, onnxruntime, sherpa-onnx,
  kokoro-onnx and websockets, under the licenses their packages state.
- PyAV (BSD-3-Clause), which reads videos. Its wheels include a build of FFmpeg (LGPL-3.0-or-later) with libx264 and
  libx265 (GPL-2.0-or-later); the sources of that build are at https://github.com/PyAV-Org/pyav-ffmpeg.
- pypdfium2 (Apache-2.0 or BSD-3-Clause), which reads PDFs, with PDFium (BSD-3-Clause).
- ggml's CPU kernels from llama.cpp (MIT).
- On Linux, the parts of AMD's ROCm the engines load (MIT and others, as ROCm states); on Windows, hipBLASLt and its
  kernels from AMD's HIP SDK (MIT).
- Bun (MIT), which runs the terminal and the web chat, and the packages in web/package.json.
- On Windows, WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos (https://github.com/winfsp/winfsp),
  as its unmodified installer (GPLv3 with its exception for free software), and `build/ramdrive.exe`, WinFsp's own
  in-memory file system sample built unchanged from that release (GPLv3; its source is `samples/memfs` in WinFsp 2.1).
  Incognito chats keep OpenCode's files on it, so they are only ever in memory.

web/THIRD-PARTY.md lists the code in web/ that is ported from other projects.
