CUDA  ?= /usr/local/cuda
NVCC  := $(CUDA)/bin/nvcc
ARCH  ?= native
FLAGS := -O3 -std=c++20 -arch=$(ARCH) -Xcompiler -fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden,-ffunction-sections,-fdata-sections,-Wno-unused-result
C     := engine/cuda
G     := engine/gpu
SRC   := $(C)/27b/relayout.cu $(C)/27b/quant.cu $(C)/27b/ops.cu $(C)/gdn.cu $(C)/27b/attn.cu $(C)/vision.cu $(G)/27b/model.cpp $(G)/27b/load.cpp \
         $(G)/27b/api.cpp
KERN  := $(filter %.cu,$(SRC))
LIBS  := -L$(CUDA)/lib64 -lcublasLt -Xlinker -rpath,$(CUDA)/lib64 -Xlinker --no-undefined -Xlinker --gc-sections

OBJ := $(SRC:%=build/o27/%.o)
build/o27/%.o: % $(wildcard engine/*.h $(C)/*.h $(C)/27b/*.h $(G)/27b/*.h)
	@mkdir -p $(@D)
	$(NVCC) $(FLAGS) -I$(C)/27b -I$(C) -I$(G)/27b -x cu -c -o $@ $<

build/libchrysalis.so: $(OBJ)
	$(NVCC) $(FLAGS) -shared -Xlinker -s -o $@ $(OBJ) $(LIBS)

build/test_quant build/test_attn build/test_gdn build/test_vision: build/%: tests/cuda/%.cu $(KERN) $(wildcard engine/*.h $(C)/*.h $(C)/27b/*.h $(G)/27b/*.h)
	@mkdir -p build
	$(NVCC) $(FLAGS) -I$(C)/27b -I$(C) -I$(G)/27b -o $@ $< $(KERN) $(LIBS)

FN_SRC := $(C)/flash-next/relayout.cu $(C)/flash-next/quant.cu $(C)/flash-next/ops.cu $(C)/gdn.cu $(C)/flash-next/attn.cu \
          $(C)/flash-next/moe.cu $(C)/flash-next/ple.cu $(C)/vision.cu $(G)/flash-next/model.cpp $(G)/flash-next/load.cpp \
          $(C)/flash-next/experts.cpp $(G)/flash-next/api.cpp
FN_CPU := $(G)/flash-next/cpu_moe.cpp engine/cpu_exact.cpp
FN_LIBS := $(LIBS) -L$(GGML)/build/bin -lggml-cpu -lggml-base -Xlinker -rpath,$(GGML)/build/bin

FN_OBJ := $(FN_SRC:%=build/ofn/%.o) $(FN_CPU:%=build/ofn/%.o)
$(FN_CPU:%=build/ofn/%.o): build/ofn/%.o: % $(wildcard engine/*.h $(C)/*.h $(C)/flash-next/*.h $(G)/flash-next/*.h)
	@mkdir -p $(@D)
	$(CXX_HOST) -O3 -std=c++20 -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections -I$(C)/flash-next -I$(C) -I$(G)/flash-next -I$(GGML)/ggml/include -c -o $@ $<
build/ofn/%.o: % $(wildcard engine/*.h $(C)/*.h $(C)/flash-next/*.h $(G)/flash-next/*.h)
	@mkdir -p $(@D)
	$(NVCC) $(FLAGS) -I$(C)/flash-next -I$(C) -I$(G)/flash-next -I$(GGML)/ggml/include -x cu -c -o $@ $<

build/libchrysalis_fn.so: $(FN_OBJ)
	$(NVCC) $(FLAGS) -shared -Xlinker -s -o $@ $(FN_OBJ) $(FN_LIBS)

FN_KOBJ := $(filter %.cu.o,$(FN_OBJ))
FN_CPUO := $(FN_CPU:%=build/ofn/%.o)
build/test_%_fn: tests/cuda/test_%_fn.cpp $(FN_KOBJ) $(FN_CPUO) $(wildcard engine/*.h $(C)/*.h $(C)/flash-next/*.h $(G)/flash-next/*.h)
	@mkdir -p build
	$(NVCC) $(FLAGS) -I$(C)/flash-next -I$(C) -I$(G)/flash-next -I$(GGML)/ggml/include -x cu -c -o $@.o $<
	$(NVCC) $(FLAGS) -o $@ $@.o $(FN_KOBJ) $(FN_CPUO) $(FN_LIBS)

build/test_linear_fn: tests/cuda/test_linear_fn.cpp $(filter-out build/ofn/$(G)/flash-next/api.cpp.o,$(FN_OBJ)) $(wildcard engine/*.h $(C)/*.h $(C)/flash-next/*.h $(G)/flash-next/*.h)
	@mkdir -p build
	$(NVCC) $(FLAGS) -I$(C)/flash-next -I$(C) -I$(G)/flash-next -I$(GGML)/ggml/include -x cu -c -o $@.o $<
	$(NVCC) $(FLAGS) -o $@ $@.o $(filter-out build/ofn/$(G)/flash-next/api.cpp.o,$(FN_OBJ)) $(FN_LIBS)

build/test_gdn_fn: tests/cuda/test_gdn.cu

build/test_cpu_exact: tests/test_cpu_exact.cpp build/ofn/engine/cpu_exact.cpp.o
	@mkdir -p build
	$(CXX_HOST) -O3 -std=c++20 -I$(GGML)/ggml/include -o $@ $< build/ofn/engine/cpu_exact.cpp.o -L$(GGML)/build/bin -lggml-base -Wl,-rpath,$(GGML)/build/bin

kernels: build/test_quant build/test_attn build/test_gdn
	./build/test_quant data/models/qwen3.8-27b/qwen3.8-27b.gguf && ./build/test_attn && ./build/test_gdn

kernels_fn: build/test_quant_fn build/test_attn_fn build/test_gdn_fn build/test_mix_fn build/test_moe_fn build/test_cpujob_fn build/test_linear_fn \
            build/test_cpu_exact
	./build/test_quant_fn && ./build/test_attn_fn && ./build/test_gdn_fn && ./build/test_mix_fn && ./build/test_moe_fn && \
	./build/test_cpujob_fn && ./build/test_linear_fn && ./build/test_cpu_exact

TEST_BUILDS := build/llama_params build/ref_logits build/ref_logits_mm
define BACKEND_TESTS
	$(PY) tests/test_requests.py
	$(PY) tests/test_snapshot.py
	$(PY) tests/test_server.py
	$(PY) tests/compare_ref.py
	$(PY) tests/test_vision.py
endef

TEST_BUILDS_FN := build/ref_logits_fn build/ref_logits_mm_fn
define BACKEND_TESTS_FN
	$(PY) tests/test_snapshot.py --use qwen3.8-flash-next
	$(PY) tests/test_server.py --use qwen3.8-flash-next
	$(PY) tests/compare_fn.py
	$(PY) tests/test_mirror_budget.py
	$(PY) tests/test_vision.py --use qwen3.8-flash-next
endef

LLAMA_INC  := -I$(GGML)/include -I$(GGML)/ggml/include
LLAMA      := -O2 -std=c++17 $(LLAMA_INC)
LLAMA_LIBS := -L$(GGML)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(GGML)/build/bin:$(CUDA)/lib64
CXX_HOST   := g++

.PHONY: kernels kernels_fn
