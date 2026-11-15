#pragma once
#include <cstddef>

bool gpu_runs();
size_t gpu_free();
void gpu_zero(void *p, size_t bytes, void *stream);
