M27      := engine/metal/27b
MFN      := engine/metal/flash-next
MTC      := $(lastword $(wildcard /private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin))
XCMETAL  := $(shell xcrun -sdk macosx metal --version >/dev/null 2>&1 && echo ok)
METALC   := $(if $(XCMETAL),xcrun -sdk macosx metal,$(MTC)/metal)
METALLIB := $(if $(XCMETAL),xcrun -sdk macosx metallib,$(MTC)/metallib)
OBJCXX   := clang++
FW       := -framework Metal -framework Foundation
GGML_REF := -I$(GGML)/ggml/include -L$(GGML)/build/bin -lggml-base -Wl,-rpath,$(GGML)/build/bin

K27   := $(wildcard $(M27)/*.metal)
SRC27 := $(M27)/mtl.mm $(M27)/load.mm $(M27)/model.mm $(M27)/api.mm
H27   := $(wildcard $(M27)/*.h) engine/api.h engine/sampler.h engine/lookup.h engine/gguf.h
F27   := -std=c++20 -O2 -fobjc-arc -I$(M27) -Iengine -Wall -Wno-unused-function
FW27  := $(FW) -framework MetalPerformanceShaders -framework QuartzCore

build/air/27b/%.air: $(M27)/%.metal $(M27)/kcommon.h $(M27)/args.h
	@mkdir -p build/air/27b
	$(METALC) -std=metal3.2 -O3 -Wall -I$(M27) -c $< -o $@

build/chrysalis.metallib: $(patsubst $(M27)/%.metal,build/air/27b/%.air,$(K27))
	$(METALLIB) $^ -o $@.tmp && mv $@.tmp $@

build/libchrysalis.dylib: $(SRC27) $(H27) build/chrysalis.metallib
	@mkdir -p build
	$(OBJCXX) $(F27) -dynamiclib -o $@.tmp $(SRC27) $(FW27) && mv $@.tmp $@

build/test_27b: tests/metal/test_27b.mm $(SRC27) $(H27) build/chrysalis.metallib
	$(OBJCXX) $(F27) -o $@ $< $(filter-out $(M27)/api.mm,$(SRC27)) $(FW27)

KFN   := $(wildcard $(MFN)/kernels/*.metal)
KHFN  := $(wildcard $(MFN)/kernels/*.h)
SRCFN := $(MFN)/engine.mm $(MFN)/experts.mm $(MFN)/pass.mm $(MFN)/api.mm
HFN   := $(wildcard $(MFN)/*.h) engine/api.h engine/sampler.h engine/lookup.h
FFN   := -O3 -std=c++20 -fobjc-arc -fPIC -Wall -Wno-unused-function -Wno-missing-braces -I.
MFLAGS := -O3 -std=metal3.2 -ffast-math -Wno-unused-variable

build/air/fn/%.air: $(MFN)/kernels/%.metal $(KHFN)
	@mkdir -p build/air/fn
	$(METALC) $(MFLAGS) -c $< -o $@

build/air/fn/prefill.air: $(MFN)/prefill.metal $(KHFN)
	@mkdir -p build/air/fn
	$(METALC) $(MFLAGS) -c $< -o $@

build/chrysalis_fn.metallib: $(patsubst $(MFN)/kernels/%.metal,build/air/fn/%.air,$(KFN)) build/air/fn/prefill.air
	$(METALLIB) $^ -o $@.tmp && mv $@.tmp $@

build/libchrysalis_fn.dylib: $(SRCFN) $(HFN) build/chrysalis_fn.metallib
	@mkdir -p build
	$(OBJCXX) $(FFN) -dynamiclib -o $@.tmp $(SRCFN) $(FW) && mv $@.tmp $@

build/pf/chrysalis_fn.metallib: build/air/fn/prefill.air
	@mkdir -p build/pf
	$(METALLIB) $^ -o $@

build/test_quant_fn build/test_kern_fn: build/%: tests/metal/%.mm $(HFN) build/chrysalis_fn.metallib
	$(OBJCXX) $(FFN) -o $@ $< $(FW) $(GGML_REF)

build/pf/test_prefill_fn: tests/metal/test_prefill_fn.mm $(MFN)/prefill.h $(HFN) build/pf/chrysalis_fn.metallib
	$(OBJCXX) $(FFN) -o $@ $< $(FW) -framework MetalPerformanceShaders $(GGML_REF)

build/bench_gemv build/bench_kern: build/%: bench/metal/%.mm $(HFN) build/chrysalis_fn.metallib
	$(OBJCXX) $(FFN) -o $@ $< $(FW) $(GGML_REF)

FN_FILE := data/models/qwen3.8-flash-next/qwen3.8-flash-next.gguf

kernels: build/test_27b
	./build/test_27b

kernels_fn: build/test_quant_fn build/test_kern_fn build/pf/test_prefill_fn
	./build/test_quant_fn $(FN_FILE) && ./build/test_kern_fn && ./build/pf/test_prefill_fn $(FN_FILE) check

TEST_BUILDS := build/ref_logits
define BACKEND_TESTS
	$(PY) tests/test_snapshot.py
	$(PY) tests/test_server.py
	$(PY) tests/compare_ref.py
endef

TEST_BUILDS_FN :=
define BACKEND_TESTS_FN
	$(PY) tests/test_snapshot.py --use qwen3.8-flash-next
	$(PY) tests/test_server.py --use qwen3.8-flash-next
	$(PY) tests/metal/compare_fn.py
endef

LLAMA_INC  := -I$(GGML)/include -I$(GGML)/ggml/include
LLAMA      := -O2 -std=c++17 $(LLAMA_INC)
LLAMA_LIBS := -L$(GGML)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(GGML)/build/bin
CXX_HOST   := clang++

.PHONY: kernels kernels_fn
