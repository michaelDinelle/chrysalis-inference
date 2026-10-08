ROCM  ?= $(CURDIR)/data/rocm
HIPCC := $(ROCM)/bin/hipcc
FLAGS := -O3 -std=c++20 -fPIC --offload-arch=gfx1100 -Wno-unused-result
SRC   := engine/quant.hip engine/ops.hip engine/gdn.hip engine/attn.hip engine/model.cpp engine/api.cpp
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

.PHONY: test
