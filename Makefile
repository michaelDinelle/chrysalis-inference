BACKEND ?= $(if $(filter Darwin,$(shell uname -s)),metal,$(if $(wildcard /dev/nvidiactl),cuda,hip))
GGML    := $(CURDIR)/data/llama.cpp
PY      := data/venv/bin/python

include engine/$(BACKEND)/build.mk

LIB := $(if $(filter metal,$(BACKEND)),dylib,so)

engines: build/libchrysalis.$(LIB) build/libchrysalis_fn.$(LIB)

fn: build/libchrysalis_fn.$(LIB)

build/vram_peak.so: bench/vram.c
	@mkdir -p build
	gcc -O2 -shared -fPIC -o $@ $< -ldl -lpthread

bench:
	$(PY) bench/bench.py run

.PHONY: engines fn bench

include tests/tests.mk
