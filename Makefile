BACKEND ?= $(if $(filter Darwin,$(shell uname -s)),metal,hip)
GGML    := $(CURDIR)/data/llama.cpp
PY      := data/venv/bin/python

include engine/$(BACKEND)/build.mk

LIB := $(if $(filter metal,$(BACKEND)),dylib,so)

engines: build/libchrysalis.$(LIB) build/libchrysalis_fn.$(LIB)

fn: build/libchrysalis_fn.$(LIB)

build/test_sampler: tests/test_sampler.cpp engine/sampler.h
	@mkdir -p build
	$(CXX_HOST) -O2 -std=c++20 -I$(GGML)/include -I$(GGML)/ggml/include -o $@ $< $(LLAMA_LIBS)

build/sampler_cases: tests/sampler_cases.cpp engine/sampler.h
	@mkdir -p build
	$(CXX_HOST) -O2 -std=c++20 -o $@ $<

test_web: build/sampler_cases
	cd web && bun test

test_backends:
	cd web && bun test --timeout 1200000 ./backends.check.ts

test_chaos:
	cd web && bun test --timeout 1800000 ./chaos.check.ts

test: test_web build/libchrysalis.$(LIB) kernels build/test_sampler build/llama_params build/ref_logits build/ref_logits_mm
	./build/test_sampler
	$(PY) tests/test_chat.py
	$(PY) tests/test_requests.py
	$(PY) tests/test_cache.py
	$(PY) tests/test_documents.py
	$(PY) tests/test_snapshot.py
	$(PY) tests/test_server.py
	$(PY) tests/compare_ref.py
	$(PY) tests/test_vision.py

test_fn: kernels_fn

test_fn_model: build/libchrysalis_fn.$(LIB) build/ref_logits_fn build/ref_logits_mm_fn
	$(PY) tests/test_snapshot.py --use qwen3.8-flash-next
	$(PY) tests/test_server.py --use qwen3.8-flash-next
	$(PY) tests/compare_fn.py
	$(PY) tests/test_vision.py --use qwen3.8-flash-next

SERVER_INC := -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP -DLLAMA_SHARED -DLLAMA_SUBPROCESS \
              -I$(GGML)/tools/server -I$(GGML)/build/tools/server -I$(GGML)/tools/mtmd -I$(GGML) -I$(GGML)/common -I$(GGML)/vendor
build/llama_params: tests/llama_params.cpp
	@mkdir -p build
	$(CXX_HOST) $(LLAMA) $(SERVER_INC) -o $@ $< -L$(GGML)/build/bin -lllama-server-impl -lllama-common -lmtmd $(LLAMA_LIBS)

build/ref_logits build/ref_logits_mm build/ref_logits_fn build/ref_logits_mm_fn: build/%: tests/%.cpp
	@mkdir -p build
	$(CXX_HOST) $(LLAMA) -o $@ $< $(LLAMA_LIBS)

test_voice:
	$(PY) tests/test_voice.py

build/vram_peak.so: bench/vram.c
	@mkdir -p build
	gcc -O2 -shared -fPIC -o $@ $< -ldl -lpthread

bench:
	$(PY) bench/bench.py run

.PHONY: engines fn test test_web test_backends test_chaos test_fn test_fn_model test_voice bench
