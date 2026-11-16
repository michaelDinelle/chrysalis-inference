#pragma once
#include <hip/hip_runtime.h>
#define gpuDeviceReset hipDeviceReset
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuError_t hipError_t
#define gpuEventBlockingSync hipEventBlockingSync
#define gpuEventCreate hipEventCreate
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDestroy hipEventDestroy
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEvent_t hipEvent_t
#define gpuFree hipFree
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuHostGetDevicePointer hipHostGetDevicePointer
#define gpuHostRegister hipHostRegister
#define gpuHostRegisterDefault hipHostRegisterDefault
#define gpuHostRegisterMapped hipHostRegisterMapped
#define gpuHostUnregister hipHostUnregister
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemsetAsync hipMemsetAsync
#define gpuSetDevice hipSetDevice
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamNonBlocking hipStreamNonBlocking
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamWaitEvent hipStreamWaitEvent
#define gpuStream_t hipStream_t
#define gpuSuccess hipSuccess
#define gpuAllocDev(p, n) CK(hipMalloc(p, n))
#define gpuMallocPinned hipHostMalloc
#define gpuFreePinned hipHostFree
#define gpuMallocMapped(pp, n) hipHostMalloc(pp, n, hipHostMallocMapped)
#define gpuMallocMappedRaw(pp, n) hipHostMalloc(pp, n, hipHostMallocMapped)
#define gpuMallocCoherent(pp, n) hipHostMalloc(pp, n, hipHostMallocCoherent | hipHostMallocMapped)
#define gpuBlasLtCreate hipblasLtCreate
#define gpuBlasLtDestroy hipblasLtDestroy
#include <hip/hip_fp16.h>
#define GPU_ERROR_FORMAT "HIP error '%s' at %s:%d"
#define gpuAlloc(p, n) CK(hipMalloc(p, n))
#define gpuFreeHost hipHostFree
#define gpuMallocHost hipHostMalloc
#define GPU_BLASLT_H <hipblaslt/hipblaslt.h>
#define gpublasLtHandle_t hipblasLtHandle_t
#define gpublasLtMatmul hipblasLtMatmul
#define GPUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define GPU_LT_MATMUL_FAILED "hipblasLtMatmul failed (%dx%dx%d)"
#define gpuHostMalloc hipHostMalloc
#define gpublasLtCreate hipblasLtCreate
#define gpuHostAlloc hipHostMalloc
#define gpuHostFree hipHostFree
#define gpuLtMatmul hipblasLtMatmul
#define GPU_LT_SUCCESS HIPBLAS_STATUS_SUCCESS
#define LT_MATMUL_FAILED "hipblasLtMatmul failed (%dx%dx%d)"
#define GPU_CK_FMT "HIP error '%s' at %s:%d"
#define gpuLtHandle_t hipblasLtHandle_t
