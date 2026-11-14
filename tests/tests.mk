kernels kernels_fn test test_fn test_fn_model: export HIP_VISIBLE_DEVICES ?= $(shell env -u HIP_VISIBLE_DEVICES bun web/gpu.ts 2>/dev/null)

build/test_sampler: tests/test_sampler.cpp engine/sampler.h
	@mkdir -p build
	$(CXX_HOST) -O2 -std=c++20 $(LLAMA_INC) -o $@ $< $(LLAMA_LIBS)

build/sampler_cases: tests/sampler_cases.cpp engine/sampler.h
	@mkdir -p build
	$(CXX_HOST) -O2 -std=c++20 -o $@ $<

test_web: build/sampler_cases
	cd web && bun tests.ts

test_backends:
	cd web && bun test --timeout 1200000 ./backends.check.ts

test: test_web build/libchrysalis.$(LIB) kernels build/test_sampler $(TEST_BUILDS)
	./build/test_sampler
	$(PY) tests/test_chat.py
	$(PY) tests/test_cache.py
	$(PY) tests/test_documents.py
	$(BACKEND_TESTS)

test_fn: kernels_fn

test_fn_model: build/libchrysalis_fn.$(LIB) $(TEST_BUILDS_FN)
	$(BACKEND_TESTS_FN)

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

.PHONY: test test_web test_backends test_fn test_fn_model test_voice
