ROCM  ?= $(CURDIR)/data/rocm
HIPCC := $(ROCM)/bin/hipcc
FLAGS := -O3 -std=c++20 -fPIC --offload-arch=gfx1100 -Wno-unused-result
SRC   := engine/27b/relayout.hip engine/27b/quant.hip engine/27b/ops.hip engine/gdn.hip engine/27b/attn.hip engine/vision.hip engine/27b/model.cpp engine/27b/load.cpp engine/27b/api.cpp
KERN  := $(filter %.hip,$(SRC))
LIBS  := -L$(ROCM)/lib -lhipblaslt -Wl,-rpath,$(ROCM)/lib -Wl,--no-undefined

build/libchrysalis.so: $(SRC) $(wildcard engine/*.h engine/27b/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -Iengine/27b -x hip -shared -o $@ $(SRC) $(LIBS)

build/%: tests/%.hip $(KERN) $(wildcard engine/*.h engine/27b/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -Iengine/27b -o $@ $< $(KERN) $(LIBS)

build/test_sampler: tests/test_sampler.cpp engine/sampler.h
	@mkdir -p build
	g++ -O2 -std=c++20 -I$(GGML)/include -I$(GGML)/ggml/include -o $@ $< $(LLAMA_LIBS)

build/sampler_cases: tests/sampler_cases.cpp engine/sampler.h
	@mkdir -p build
	g++ -O2 -std=c++20 -o $@ $<

test_web: build/sampler_cases
	cd web && bun test

test_backends:
	cd web && bun test --timeout 1200000 ./backends.check.ts

test_chaos:
	cd web && bun test --timeout 1800000 ./chaos.check.ts

test: test_web build/libchrysalis.so build/test_quant build/test_attn build/test_gdn build/test_sampler build/llama_params build/ref_logits build/ref_logits_mm
	./build/test_quant data/models/chrysalis-mix.gguf && ./build/test_attn && ./build/test_gdn && ./build/test_sampler
	data/venv/bin/python tests/test_chat.py
	data/venv/bin/python tests/test_requests.py
	data/venv/bin/python tests/test_cache.py
	data/venv/bin/python tests/test_documents.py
	data/venv/bin/python tests/test_snapshot.py
	data/venv/bin/python tests/test_server.py
	data/venv/bin/python tests/compare_ref.py
	data/venv/bin/python tests/test_vision.py

GGML  := $(CURDIR)/data/llama.cpp
FN_SRC := engine/flash-next/relayout.hip engine/flash-next/quant.hip engine/flash-next/ops.hip engine/gdn.hip engine/flash-next/attn.hip engine/flash-next/moe.hip \
          engine/flash-next/ple.hip engine/vision.hip engine/flash-next/model.cpp engine/flash-next/load.cpp engine/flash-next/experts.cpp engine/flash-next/api.cpp \
          engine/flash-next/cpu_moe.cpp engine/flash-next/cpu_vnni.cpp
FN_LIBS := $(LIBS) -L$(GGML)/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$(GGML)/build/bin

build/libchrysalis_fn.so: $(FN_SRC) $(wildcard engine/*.h engine/flash-next/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -Iengine/flash-next -I$(GGML)/ggml/include -x hip -shared -o $@ $(FN_SRC) $(FN_LIBS)

fn: build/libchrysalis_fn.so

FN_KERN := $(filter-out engine/flash-next/model.cpp engine/flash-next/load.cpp engine/flash-next/experts.cpp engine/flash-next/api.cpp,$(FN_SRC))
build/test_%_fn: tests/test_%_fn.cpp $(FN_KERN) $(wildcard engine/flash-next/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -Iengine/flash-next -I$(GGML)/ggml/include -x hip -o $@ $< $(FN_KERN) $(FN_LIBS)

build/test_linear_fn: tests/test_linear_fn.cpp $(FN_SRC) $(wildcard engine/flash-next/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -Iengine/flash-next -I$(GGML)/ggml/include -x hip -o $@ $< $(filter-out engine/flash-next/api.cpp,$(FN_SRC)) $(FN_LIBS)

build/test_gdn_fn: tests/test_gdn.hip

test_fn: build/test_quant_fn build/test_attn_fn build/test_gdn_fn build/test_mix_fn build/test_moe_fn build/test_cpujob_fn \
         build/test_linear_fn
	./build/test_quant_fn && ./build/test_attn_fn && ./build/test_gdn_fn && ./build/test_mix_fn && ./build/test_moe_fn && \
	./build/test_cpujob_fn && ./build/test_linear_fn

test_fn_model: build/libchrysalis_fn.so build/ref_logits_fn build/ref_logits_mm_fn
	data/venv/bin/python tests/test_snapshot.py --use qwen3.8-flash-next
	data/venv/bin/python tests/test_server.py --use qwen3.8-flash-next
	data/venv/bin/python tests/compare_fn.py
	data/venv/bin/python tests/test_vision.py --use qwen3.8-flash-next

LLAMA      := -O2 -std=c++17 -I$(GGML)/include -I$(GGML)/ggml/include
LLAMA_LIBS := -L$(GGML)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(GGML)/build/bin:$(ROCM)/lib
SERVER_INC := -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP -DLLAMA_SHARED -DLLAMA_SUBPROCESS \
              -I$(GGML)/tools/server -I$(GGML)/build/tools/server -I$(GGML)/tools/mtmd -I$(GGML) -I$(GGML)/common -I$(GGML)/vendor
build/llama_params: tests/llama_params.cpp
	@mkdir -p build
	g++ $(LLAMA) $(SERVER_INC) -o $@ $< -L$(GGML)/build/bin -lllama-server-impl -lllama-common -lmtmd $(LLAMA_LIBS)

build/ref_logits: tests/ref_logits.cpp
	@mkdir -p build
	g++ $(LLAMA) -o $@ $< $(LLAMA_LIBS)

build/ref_logits_mm: tests/ref_logits_mm.cpp
	@mkdir -p build
	g++ $(LLAMA) -o $@ $< $(LLAMA_LIBS)

build/ref_logits_fn: tests/ref_logits_fn.cpp
	@mkdir -p build
	g++ $(LLAMA) -o $@ $< $(LLAMA_LIBS)

build/ref_logits_mm_fn: tests/ref_logits_mm_fn.cpp
	@mkdir -p build
	g++ $(LLAMA) -o $@ $< $(LLAMA_LIBS)

test_voice:
	data/venv/bin/python tests/test_voice.py

.PHONY: test_voice

build/vram_peak.so: bench/vram.c
	@mkdir -p build
	gcc -O2 -shared -fPIC -o $@ $< -ldl -lpthread

bench:
	data/venv/bin/python bench/bench.py run

.PHONY: bench

.PHONY: test test_web test_backends test_chaos fn test_fn test_fn_model
