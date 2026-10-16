ROCM  ?= $(CURDIR)/data/rocm
HIPCC := $(ROCM)/bin/hipcc
FLAGS := -O3 -std=c++20 -fPIC --offload-arch=gfx1100 -Wno-unused-result
SRC   := engine/relayout.hip engine/quant.hip engine/ops.hip engine/gdn.hip engine/attn.hip engine/model.cpp engine/load.cpp engine/api.cpp
KERN  := $(filter %.hip,$(SRC))
LIBS  := -L$(ROCM)/lib -lhipblaslt -Wl,-rpath,$(ROCM)/lib

build/libkiln.so: $(SRC) $(wildcard engine/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -x hip -shared -o $@ $(SRC) $(LIBS)

# kernel tests and benchmarks: build/<name> from tools/<name>.hip
build/%: tools/%.hip $(KERN) $(wildcard engine/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -o $@ $< $(KERN) $(LIBS)

test: build/libkiln.so build/test_quant build/test_attn
	./build/test_quant data/models/kiln-mix.gguf && ./build/test_attn
	data/venv/bin/python tools/test_chat.py
	data/venv/bin/python tools/test_snapshot.py
	data/venv/bin/python tools/test_server.py

# Qwen3.8-Flash-Next engine (engine_fn/): its cold experts run on the CPU with ggml's kernels from data/llama.cpp
GGML  := $(CURDIR)/data/llama.cpp
FN_SRC := engine_fn/relayout.hip engine_fn/quant.hip engine_fn/ops.hip engine_fn/gdn.hip engine_fn/attn.hip engine_fn/moe.hip \
          engine_fn/ple.hip engine_fn/model.cpp engine_fn/load.cpp engine_fn/api.cpp engine_fn/cpu_moe.cpp
FN_LIBS := $(LIBS) -L$(GGML)/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$(GGML)/build/bin

build/libkiln_fn.so: $(FN_SRC) $(wildcard engine_fn/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -I$(GGML)/ggml/include -x hip -shared -o $@ $(FN_SRC) $(FN_LIBS)

fn: build/libkiln_fn.so

.PHONY: test fn
