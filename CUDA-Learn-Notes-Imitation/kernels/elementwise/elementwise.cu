#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>      // torch::Tensor  

#define WARP_SIZE 32
//宏定义, 将value的数据类型强制转换为int4类型。
// reinterpret_cast<int4*>(&(value)) 将 value的地址转换为int4类型的指针
// reinterpret_cast<int4*>(&(value))[0] 取出指针指向的第一个元素 
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0]) 
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])


// -------------------------------------- FP32 --------------------------------------
// ElementWise Add
// grid(N/256), block(256)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
// 256个线程块, 每个线程块256个线程, 平铺为一维度
__global__ void elementwise_add_f32_kernel(float* a, float* b, float* c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];  // 防止越界访问
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
// 256个线程块, 每个线程块256/4=64个线程, 平铺为一维度
// 每个线程 使用 Vec4 处理四个数据
// 
__global__ void elementwise_add_f32x4_kernel(float* a, float* b, float* c, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        // FLOAT4(a[idx])：通过宏 FLOAT4 将 a[idx] 的地址 reinterpret 为 float4 类型，并加载 4 个连续的 float 数据到寄存器 reg_a。
        // float4 是 CUDA 提供的向量类型，包含 4 个 float 元素（x, y, z, w）。       
        float4 reg_a = FLOAT4(a[idx]); 
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c; // 在 寄存器中创建
        /*
        // 合并写法
        float4 reg_c = reg_a + reg_b; 
        FLOAT4(c[idx]) = reg_c; // 将 reg_c 中的 128 位数据（即 4 个连续的 float 值）存储到 c[idx] 的地址中。
        */
        // 逐分量写法   
        reg_c.x = reg_a.x + reg_b.x; // 将 reg_a 和 reg_b 中的每个分量相加
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c; // 将 reg_c 中的 128 位数据（即 4 个连续的 float 值）存储到 c[idx] 的地址中。
    }
}


// -------------------------------------- FP16 --------------------------------------

