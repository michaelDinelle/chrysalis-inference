#pragma once
#include <cuda_runtime.h>
#define gpuDeviceReset cudaDeviceReset
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuError_t cudaError_t
#define gpuEventBlockingSync cudaEventBlockingSync
#define gpuEventCreate cudaEventCreate
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDestroy cudaEventDestroy
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEvent_t cudaEvent_t
#define gpuFree cudaFree
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuHostGetDevicePointer cudaHostGetDevicePointer
#define gpuHostRegister cudaHostRegister
#define gpuHostRegisterDefault cudaHostRegisterDefault
#define gpuHostRegisterMapped cudaHostRegisterMapped
#define gpuHostUnregister cudaHostUnregister
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemsetAsync cudaMemsetAsync
#define gpuSetDevice cudaSetDevice
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamWaitEvent cudaStreamWaitEvent
#define gpuStream_t cudaStream_t
#define gpuSuccess cudaSuccess
#define gpuAllocDev(p, n) zalloc(p, n)
#define gpuMallocPinned cudaMallocHost
#define gpuFreePinned cudaFreeHost
#define gpuMallocMapped(pp, n) cudaHostAlloc((void **)pp, n, cudaHostAllocMapped)
#define gpuMallocMappedRaw(pp, n) cudaHostAlloc(pp, n, cudaHostAllocMapped)
#define gpuMallocCoherent(pp, n) cudaHostAlloc((void **)pp, n, cudaHostAllocMapped)
#define gpuBlasLtCreate cublasLtCreate
#define gpuBlasLtDestroy cublasLtDestroy
#include <cuda_fp16.h>
#define GPU_ERROR_FORMAT "CUDA error '%s' at %s:%d"
#define gpuAlloc(p, n) zalloc(p, n)
#define gpuFreeHost cudaFreeHost
#define gpuMallocHost cudaMallocHost
#define GPU_BLASLT_H <cublasLt.h>
#define gpublasLtHandle_t cublasLtHandle_t
#define gpublasLtMatmul cublasLtMatmul
#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define GPU_LT_MATMUL_FAILED "cublasLtMatmul failed (%dx%dx%d)"
#define gpuHostMalloc cudaMallocHost
#define gpublasLtCreate cublasLtCreate
#define gpuHostAlloc cudaMallocHost
#define gpuHostFree cudaFreeHost
#define gpuLtMatmul cublasLtMatmul
#define GPU_LT_SUCCESS CUBLAS_STATUS_SUCCESS
#define LT_MATMUL_FAILED "cublasLtMatmul failed (%dx%dx%d)"
#define GPU_CK_FMT "CUDA error '%s' at %s:%d"
#define gpuLtHandle_t cublasLtHandle_t
