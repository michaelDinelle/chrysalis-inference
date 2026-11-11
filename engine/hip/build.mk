ROCM  ?= $(CURDIR)/data/rocm
HIPCC := $(ROCM)/bin/hipcc
FLAGS := -O3 -std=c++20 -fPIC --offload-arch=gfx1100 -Wno-unused-result -ffunction-sections -fdata-sections -fvisibility=hidden -fvisibility-inlines-hidden
H     := engine/hip
SRC   := $(H)/27b/relayout.hip $(H)/27b/quant.hip $(H)/27b/ops.hip $(H)/gdn.hip $(H)/27b/attn.hip $(H)/vision.hip $(H)/27b/model.cpp $(H)/27b/load.cpp \
         $(H)/27b/api.cpp
KERN  := $(filter %.hip,$(SRC))
LIBS  := -L$(ROCM)/lib -lhipblaslt -Wl,-rpath,$(ROCM)/lib -Wl,--no-undefined -Wl,--gc-sections

OBJ := $(SRC:%=build/o27/%.o)
build/o27/%.o: % $(wildcard engine/*.h $(H)/*.h $(H)/27b/*.h)
	@mkdir -p $(@D)
	$(HIPCC) $(FLAGS) -I$(H)/27b -x hip -c -o $@ $<
build/o27/%.cpp.o: %.cpp $(wildcard engine/*.h $(H)/*.h $(H)/27b/*.h)
	@mkdir -p $(@D)
	$(HIPCC) $(FLAGS:-O3=-Os) -I$(H)/27b -x hip -c -o $@ $<

build/libchrysalis.so: $(OBJ)
	$(HIPCC) $(FLAGS) -shared -s -o $@ $(OBJ) $(LIBS)

build/test_quant build/test_attn build/test_gdn build/test_vision: build/%: tests/hip/%.hip $(KERN) $(wildcard engine/*.h $(H)/*.h $(H)/27b/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -I$(H)/27b -o $@ $< $(KERN) $(LIBS)

FN_SRC := $(H)/flash-next/relayout.hip $(H)/flash-next/quant.hip $(H)/flash-next/ops.hip $(H)/gdn.hip $(H)/flash-next/attn.hip \
          $(H)/flash-next/moe.hip $(H)/flash-next/ple.hip $(H)/vision.hip $(H)/flash-next/model.cpp $(H)/flash-next/load.cpp \
          $(H)/flash-next/experts.cpp $(H)/flash-next/api.cpp $(H)/flash-next/cpu_moe.cpp $(H)/flash-next/cpu_vnni.cpp
FN_LIBS := $(LIBS) -L$(GGML)/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$(GGML)/build/bin

FN_OBJ := $(FN_SRC:%=build/ofn/%.o)
build/ofn/%.o: % $(wildcard engine/*.h $(H)/*.h $(H)/flash-next/*.h)
	@mkdir -p $(@D)
	$(HIPCC) $(FLAGS) -I$(H)/flash-next -I$(GGML)/ggml/include -x hip -c -o $@ $<

build/libchrysalis_fn.so: $(FN_OBJ)
	$(HIPCC) $(FLAGS) -shared -s -o $@ $(FN_OBJ) $(FN_LIBS)

FN_KERN := $(filter-out $(H)/flash-next/model.cpp $(H)/flash-next/load.cpp $(H)/flash-next/experts.cpp $(H)/flash-next/api.cpp,$(FN_SRC))
build/test_%_fn: tests/hip/test_%_fn.cpp $(FN_KERN) $(wildcard $(H)/flash-next/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -I$(H)/flash-next -I$(GGML)/ggml/include -x hip -o $@ $< $(FN_KERN) $(FN_LIBS)

build/test_linear_fn: tests/hip/test_linear_fn.cpp $(FN_SRC) $(wildcard $(H)/flash-next/*.h)
	@mkdir -p build
	$(HIPCC) $(FLAGS) -I$(H)/flash-next -I$(GGML)/ggml/include -x hip -o $@ $< $(filter-out $(H)/flash-next/api.cpp,$(FN_SRC)) $(FN_LIBS)

build/test_gdn_fn: tests/hip/test_gdn.hip

kernels: build/test_quant build/test_attn build/test_gdn
	./build/test_quant data/models/qwen3.8-27b/qwen3.8-27b.gguf && ./build/test_attn && ./build/test_gdn

kernels_fn: build/test_quant_fn build/test_attn_fn build/test_gdn_fn build/test_mix_fn build/test_moe_fn build/test_cpujob_fn build/test_linear_fn
	./build/test_quant_fn && ./build/test_attn_fn && ./build/test_gdn_fn && ./build/test_mix_fn && ./build/test_moe_fn && \
	./build/test_cpujob_fn && ./build/test_linear_fn

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
	$(PY) tests/test_vision.py --use qwen3.8-flash-next
endef

LLAMA_INC  := -I$(GGML)/include -I$(GGML)/ggml/include
LLAMA      := -O2 -std=c++17 $(LLAMA_INC)
LLAMA_LIBS := -L$(GGML)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(GGML)/build/bin:$(ROCM)/lib
CXX_HOST   := g++

.PHONY: kernels kernels_fn
