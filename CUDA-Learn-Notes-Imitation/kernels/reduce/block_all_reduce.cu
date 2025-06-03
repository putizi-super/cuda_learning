#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])



// ---- FP32 -----
// Warp Reduce Sum

// ————shfl_xor_sync  在warp内线程之间交换数据
// ————shfl_down_sync 在warp内同一线程交换数据
// xffffffff 线程掩码，表示warp中素有线程都参与
// mask 控制线程之间的数据交换  步长

// warp_reduce_sum_fp32 函数用于在一个warp内对浮点数进行归约求和操作
// 该函数使用了CUDA的线程间通信函数__shfl_xor_sync来实现线程之间的数据交换
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_fp32(float val){
    #pragma unroll // 指示编译器将循环展开（unroll）， 以提高性能
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}


// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y = sum(a)
/*

- NUM_THREADS 是一个模版参数，用于定义 CUDA 内核中每个线程块的线程数量，是一个编译时常量，默认值为256 
- 通过使用模版参数 NUM_THREADS, 可以灵活地调整线程块的大小，而无需在代码中硬编码具体的线程数量。

在调用这个内核函数时，可以使用默认的 NUM_THREADS = 256, 那么每个线程块都会有 256 个线程。也可以调用时指定其他值：

block_all_reduce_sum_fp32_kernel<128><<<gridDim, blockDim>>>(a, y, N);

*/
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp32_kernel(float* a, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE; // 计算每个block中的warp数量
    __shared__ float reduce_smem[NUM_WARPS]; // 每个block 共用一个 SM, 将结果保存到 SM 中
    
    // 将 数据加载到寄存器中
    float sum = (idx < N) ? a[idx] : 0.0f; // 读取数据

    int warp = tid / WARP_SIZE; // 计算当前线程所在的warp编号, 由于 threadIdx 是 一个 block 内的线程索引，所以可以通过除以 WARP_SIZE 来计算当前线程所在的 warp 编号
    int lane = tid % WARP_SIZE; // 计算当前线程在warp中的索引

    // 在warp内进行归约
    sum = warp_reduce_sum_fp32<WARP_SIZE>(sum); // 在warp内进行归约求和
    // 将每个warp的结果存储到共享内存中
    if (lane == 0){
        reduce_smem[warp] = sum; // 将每个warp的结果存储到共享内存中
    }
    __syncthreads(); // 同步所有线程，确保所有warp的结果都存储到共享内存中, 确保整个 block 进行规约时 可以实现
    // 在block内进行归约
    
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f; // 读取共享内存中的结果

    if(warp == 0) sum = warp_reduce_sum_fp32<WARP_SIZE>(sum); // 在第一个warp内进行归约求和
    if (tid == 0) atomicAdd(y, sum); // 将结果写入全局内存
}

// Block All Reduce Sum + float4 
// grid(N/256), block(256/4)
// a: Nx1, y = sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float* a, float* y, int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE; // 计算每个block中的warp数量


}
