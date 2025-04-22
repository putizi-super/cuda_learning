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
#include <torch/extension.h> // torch::Tensor

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
// ElementWise Add
// grid(N/256), block(256)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half* a, half* b, half* c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = __hadd(a[idx], b[idx]); // __hadd 是 CUDA 提供的半精度浮点数加法函数
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half* a, half* b, half* c, int N){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_a = HALF2(a[idx]); // 将 a[idx] 的地址 reinterpret 为 half2 类型，并加载 2 个连续的 half 数据到寄存器 reg_a。
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c; // 在 寄存器中创建
        /*
        // 合并写法
        half2 reg_c = __hadd(reg_a, reg_b); 
        HALF2(c[idx]) = reg_c; // 将 reg_c 中的 64 位数据（即 2 个连续的 half 值）存储到 c[idx] 的地址中。
        */
        // 逐分量写法   
        reg_c.x = __hadd(reg_a.x, reg_b.x); // 将 reg_a 和 reg_b 中的每个分量相加
        reg_c.y = __hadd(reg_a.y, reg_b.y); // __hadd 是 CUDA 提供的一个 半精度浮点数加法函数，用于对两个 half 类型的变量执行加法操作
        HALF2(c[idx]) = reg_c; // 将 reg_c 中的 64 位数据（即 2 个连续的 half 值）存储到 c[idx] 的地址中。
    }
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x8_kernel(half* a, half* b, half* c, int N){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_a_0 = HALF2(a[idx + 0]); // 将 a[idx] 的地址 reinterpret 为 half2 类型，并加载 2 个连续的 half 数据到寄存器 reg_a。
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);
    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3; // 在 寄存器中创建
    /*
    // 合并写法
    half2 reg_c_0 = __hadd(reg_a_0, reg_b_0); 
    half2 reg_c_1 = __hadd(reg_a_1, reg_b_1); 
    half2 reg_c_2 = __hadd(reg_a_2, reg_b_2); 
    half2 reg_c_3 = __hadd(reg_a_3, reg_b_3); 
    HALF4(c[idx]) = {reg_c_0, reg_c_1, reg_c_2, reg_c_3}; // 将四个半精度浮点数存储到 c[idx] 的地址中。
    */
    // 逐分量写法   
    // 将每个分量相加
    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
    // 将结果存储到 c[idx] 的地址中
    if ((idx + 0) < N) { HALF2(c[idx + 0]) = reg_c_0; }
    if ((idx + 2) < N) { HALF2(c[idx + 2]) = reg_c_1; } // 防止越界访问    
    if ((idx + 4) < N) { HALF2(c[idx + 4]) = reg_c_2; }
    if ((idx + 6) < N) { HALF2(c[idx + 6]) = reg_c_3; }
}


// 以下函数通过将数值 pack 到列表中, 使用循环操作
__global__ void elementwise_add_f16x8_pack_kernel(half* a, half* b, half* c, int N){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    // 会优先存储在 RAG中, 当寄存器不够时, 才会存储在L2 Cache Local Memeory中,在 ptx 中叫做 local space, 是可以寻址的
    half pack_a[8], pack_b[8], pack_c[8]; // 8 x 16 = 128 bits
    // 强制转换为 float4 类型，并加载 128 位数据到 pack_a 和 pack_b 中
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits

    #pragma unroll // 编译器指令，告诉编译器在编译时展开循环, 好处是可以减少循环控制开销

    for (int i = 0; i < 8; i++) {
        pack_c[i] = __hadd(pack_a[i], pack_b[i]);
    }
    LDST128BITS(c[idx]) = LDST128BITS(pack_c); // store 128 bits

    // 将输出强制转换为 float4 类型，并存储到 c[idx] 的地址中
    if((idx + 7) < N) { LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]); }
}
/*
该方法更快的原因：
- 内存访问模式
  - elementwise_add_f16x8_kernel 的内存访问,逐个叫在数据，每次加载指令分散的。
  - elementwise_add_f16x8_pack_kernel 的内存访问，使用 128位的加载指令，128位的加载指令可以在一个内存访问中加载多个数据。属于向量化加载,可以充分利用GPU的内存带宽,减少内存访问指令的数量。
- 循环展开与寄存器优化
  - elementwise_add_f16x8_kernel：手动展开了 4 次 half2 的加法操作（reg_a_0, reg_a_1, reg_a_2, reg_a_3）。每次操作都需要单独的寄存器存储中间结果，寄存器使用量较高。
  - elementwise_add_f16x8_pack_kernel：使用了一个局部数组 pack_a 和 pack_b，并通过循环处理 8 个 half 数据。
    使用 #pragma unroll 指令展开循环，编译器会自动优化寄存器分配和指令调度。
    循环展开后，寄存器使用量更低，指令调度更高效。
*/

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str // 将 str 转换为字符串 #str是一个预处理器指令，将 str 转换为字符串字面量
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func)); // 定义一个函数，将 func 转换为字符串，并将其作为函数名，将 func 作为函数体，将 func 转换为字符串，并将其作为函数名，将 func 作为函数体
// 宏 TORCH_BINDING_COMMON_EXTENSION 接受一个参数 func，表示要绑定的函数。
// 宏展开后会调用 m.def，将函数 func 绑定到 PyTorch 的 Python 接口中。
/*
m.def 是 PyTorch 的 C++ 扩展 API，用于将 C++ 函数绑定到 Python。
语法：
m.def("python_name", &cpp_function, "docstring");
- "python_name"：在 Python 中调用该函数时使用的名称。
- &cpp_function：C++ 中实现该函数的指针。
- "docstring"：函数的文档字符串（可选）。
*/

// 检查 PyTorch Tensor 的数据类型
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

/**
 * @brief 执行两个张量的逐元素加法，并将结果存储到第三个张量中。
 *
 * 该宏定义了一个用于特定数据类型的张量逐元素加法函数。
 * 函数会检查输入张量的数据类型，确定适当的 CUDA kernel 启动配置，
 * 并调用相应的 kernel 进行计算。
 *
 * @param a 输入张量 `a`，类型为 `torch::Tensor`。必须具有指定的数据类型。
 * @param b 输入张量 `b`，类型为 `torch::Tensor`。必须具有指定的数据类型。
 * @param c 输出张量 `c`，类型为 `torch::Tensor`。必须具有指定的数据类型。
 *
 * 该函数支持任意维度的张量。如果张量的维度不为 2，则会计算总元素数 N，
 * 并根据 N 设置 CUDA kernel 的网格和线程块配置。
 * 如果张量的维度为 2，则会根据张量的形状（S 和 K）设置不同的 kernel 配置。
 *
 * CUDA kernel 的配置会根据张量的大小和元素类型的数量动态调整，以确保高效的计算。
 */
 #define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements)   \
 void elementwise_add_##packed_type(                                              \
   torch::Tensor a, torch::Tensor b, torch::Tensor c) {                           \
   CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                         \
   CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                         \
   CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                         \
   const int ndim = a.dim();                                                      \
   if (ndim != 2) {                                                               \
     int N = 1;                                                                   \
     for (int i = 0; i < ndim; ++i) { N *= a.size(i); }                           \
     dim3 block(256 / (n_elements));                                              \
     dim3 grid((N + 256 - 1) / 256);                                              \
     elementwise_add_##packed_type##_kernel<<<grid, block>>>(                     \
       reinterpret_cast<element_type*>(a.data_ptr()),                             \
       reinterpret_cast<element_type*>(b.data_ptr()),                             \
       reinterpret_cast<element_type*>(c.data_ptr()), N);                         \
   } else {                                                                       \
     const int S = a.size(0);                                                     \
     const int K = a.size(1);                                                     \
     const int N = S * K;                                                         \
     if ((K/(n_elements)) <= 1024) {                                              \
       dim3 block(K/(n_elements));                                                \
       dim3 grid(S);                                                              \
       elementwise_add_##packed_type##_kernel<<<grid, block>>>(                   \
         reinterpret_cast<element_type*>(a.data_ptr()),                           \
         reinterpret_cast<element_type*>(b.data_ptr()),                           \
         reinterpret_cast<element_type*>(c.data_ptr()), N);                       \
     } else {                                                                     \
       int N = 1;                                                                 \
       for (int i = 0; i < ndim; ++i) { N *= a.size(i); }                         \
       dim3 block(256 / (n_elements));                                            \
       dim3 grid((N + 256 - 1) / 256);                                            \
       elementwise_add_##packed_type##_kernel<<<grid, block>>>(                   \
         reinterpret_cast<element_type*>(a.data_ptr()),                           \
         reinterpret_cast<element_type*>(b.data_ptr()),                           \
         reinterpret_cast<element_type*>(c.data_ptr()), N);                       \
     }                                                                            \
   }                                                                              \
 }
 
 

TORCH_BINDING_ELEM_ADD(f32,         torch::kFloat32,    float,    1)
TORCH_BINDING_ELEM_ADD(f32x4,       torch::kFloat32,    float,    4)
TORCH_BINDING_ELEM_ADD(f16,         torch::kHalf,       half,     1)
TORCH_BINDING_ELEM_ADD(f16x2,       torch::kHalf,       half,     2)
TORCH_BINDING_ELEM_ADD(f16x8,       torch::kHalf,       half,     8)
TORCH_BINDING_ELEM_ADD(f16x8_pack,  torch::kHalf,       half,     8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}
// 以上代码定义了一个用于 PyTorch 的 C++ 扩展模块