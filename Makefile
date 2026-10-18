ROCM  ?= $(CURDIR)/data/rocm
HIPCC := $(ROCM)/bin/hipcc
FLAGS := -O3 -std=c++20 -fPIC --offload-arch=gfx1100 -Wno-unused-result
SRC   := engine/relayout.hip engine/quant.hip engine/ops.hip engine/gdn.hip engine/attn.hip engine/model.cpp engine/load.cpp engine/api.cpp
KERN  := $(filter %.hip,$(SRC))
LIBS  := -L$(ROCM)/lib -lhipblaslt -Wl,-rpath,$(ROCM)/lib -Wl,--no-undefined

build/libkiln.so: $(SRC) $(wildcard engine/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -x hip -shared -o $@ $(SRC) $(LIBS)

# kernel tests and benchmarks: build/<name> from tools/<name>.hip
build/%: tools/%.hip $(KERN) $(wildcard engine/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -o $@ $< $(KERN) $(LIBS)

build/test_sampler: tools/test_sampler.cpp engine/sampler.h
	@mkdir -p build
	g++ -O2 -std=c++20 -o $@ $<

build/sampler_cases: tools/sampler_cases.cpp engine/sampler.h
	@mkdir -p build
	g++ -O2 -std=c++20 -o $@ $<

# the web UI (no GPU): its copy of the sampler against the engine's, the server's replies against a fake kiln
test_web: build/sampler_cases
	cd web && bun test

test: test_web build/libkiln.so build/test_quant build/test_attn build/test_gdn build/test_sampler build/ref_logits
	./build/test_quant data/models/kiln-mix.gguf && ./build/test_attn && ./build/test_gdn && ./build/test_sampler
	data/venv/bin/python tools/test_chat.py
	data/venv/bin/python tools/test_cache.py
	data/venv/bin/python tools/test_snapshot.py
	data/venv/bin/python tools/test_server.py
	data/venv/bin/python tools/compare_ref.py

# Qwen3.8-Flash-Next engine (engine_fn/): its cold experts run on the CPU with ggml's kernels from data/llama.cpp
GGML  := $(CURDIR)/data/llama.cpp
FN_SRC := engine_fn/relayout.hip engine_fn/quant.hip engine_fn/ops.hip engine_fn/gdn.hip engine_fn/attn.hip engine_fn/moe.hip \
          engine_fn/ple.hip engine_fn/model.cpp engine_fn/load.cpp engine_fn/api.cpp engine_fn/cpu_moe.cpp engine_fn/cpu_vnni.cpp
FN_LIBS := $(LIBS) -L$(GGML)/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$(GGML)/build/bin

build/libkiln_fn.so: $(FN_SRC) $(wildcard engine_fn/*.h) engine/sampler.h
	@mkdir -p build
	$(HIPCC) $(FLAGS) -I$(GGML)/ggml/include -x hip -shared -o $@ $(FN_SRC) $(FN_LIBS)

fn: build/libkiln_fn.so

# its kernel tests: build_fn/<name> from tools_fn/<name>.cpp with the kernels and the CPU expert code
FN_KERN := $(filter-out engine_fn/model.cpp engine_fn/load.cpp engine_fn/api.cpp,$(FN_SRC))
build_fn/test_%_fn: tools_fn/test_%_fn.cpp $(FN_KERN) $(wildcard engine_fn/*.h)
	@mkdir -p build_fn
	$(HIPCC) $(FLAGS) -I$(GGML)/ggml/include -x hip -o $@ $< $(FN_KERN) $(FN_LIBS)

build_fn/bench_%_fn: tools_fn/bench_%_fn.cpp $(FN_KERN) $(wildcard engine_fn/*.h)
	@mkdir -p build_fn
	$(HIPCC) $(FLAGS) -I$(GGML)/ggml/include -x hip -o $@ $< $(FN_KERN) $(FN_LIBS)

# the engine's own GEMM path (Engine::linear) needs the whole engine
build_fn/test_linear_fn: tools_fn/test_linear_fn.cpp $(FN_SRC) $(wildcard engine_fn/*.h)
	@mkdir -p build_fn
	$(HIPCC) $(FLAGS) -I$(GGML)/ggml/include -x hip -o $@ $< $(filter-out engine_fn/api.cpp,$(FN_SRC)) $(FN_LIBS)

build_fn/test_gdn_fn: tools/test_gdn.hip

test_fn: build_fn/test_quant_fn build_fn/test_attn_fn build_fn/test_gdn_fn build_fn/test_mix_fn build_fn/test_moe_fn build_fn/test_cpujob_fn \
         build_fn/test_linear_fn
	./build_fn/test_quant_fn && ./build_fn/test_attn_fn && ./build_fn/test_gdn_fn && ./build_fn/test_mix_fn && ./build_fn/test_moe_fn && \
	./build_fn/test_cpujob_fn && ./build_fn/test_linear_fn

# the Flash-Next model in-process (kiln stopped, ~47 GiB of system RAM): generation and snapshots, the server, and its
# next-token distributions against llama.cpp's
test_fn_model: build/libkiln_fn.so build_fn/ref_logits_fn
	data/venv/bin/python tools/test_snapshot.py --use qwen3.8-flash-next
	data/venv/bin/python tools/test_server.py --use qwen3.8-flash-next
	data/venv/bin/python tools_fn/compare_fn.py

# llama.cpp's logits for those comparisons (tools/compare_ref.py, tools_fn/compare_fn.py)
LLAMA      := -O2 -std=c++17 -I$(GGML)/include -I$(GGML)/ggml/include
LLAMA_LIBS := -L$(GGML)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(GGML)/build/bin:$(ROCM)/lib
build/ref_logits: tools/ref_logits.cpp
	@mkdir -p build
	g++ $(LLAMA) -o $@ $< $(LLAMA_LIBS)

build_fn/ref_logits_fn: tools_fn/ref_logits_fn.cpp
	@mkdir -p build_fn
	g++ $(LLAMA) -o $@ $< $(LLAMA_LIBS)

# the mechanical cases of AGENTS.md's comment rule
lint:
	python3 -I tools/lint_comments.py

.PHONY: test test_web fn test_fn test_fn_model lint
