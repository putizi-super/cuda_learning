#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>


#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&value)[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&value)[0])
#define HALF2(value) (reinterpret_cast<half2*>(&value)[0])
#define BFLOAT16(value) (reinterpret_cast<bfloat16*>(&value)[0])
#define LDST128BITS(value) (reinterpret_cast<long long*>(&value)[0])

// 定义全局 alpah值
#define ALPHA 1.0f 



// T.options() 是 pytorch 里的一个方法，用于获取张量(Tensor)的配置信息，具体包括返回一个 TensorOptions 对象，包括张量的以下属性：dtype（数据类型）、device（设备）、layout（布局）

// ELU 计算函数
// ---------- FP32 ----------
// CUDA 函数类型限定符，表示该函数只能在GPU 设备上执行
// __forceinline__ 强制内联
// __inline__ 建议内联
/*
关于内联：
内联函数是指编译器将函数体插入到调用该函数的地方，而不是将函数调用作为单独的指令执行。
比如普通函数调用：
float result = elu(x); // 需要函数调用开销
内联之后：
float result = x > 0 ? x : ALPHA * (expf(x) - 1.f); // 编译器会将函数体直接展开在调用处
优点： 1. 消除函数调用开销 2. 减少栈操作 3. 提高执行速度 4. 便于编译器优化
缺点： 1. 增加大妈体积 2. 过度使用导致程序膨胀

建议内联，编译器会根据选择忽略内联请求。
*/ 
__device__ __forceinline__ float elu(float x){
    return x > 0 ? x : ALPHA * (expf(x) - 1.f);
}

// ---------- FP16 ----------

/*
__hgt() 是 CUDA 中用于 FP16 到比较运算函数
1. __hgt() = half greater than
2. 用于比较两个半精度浮点数到大小
3. 返回布尔值
类似还有： __hlt() __heq() __hge() __hle()

为什么 half 不使用 >:
> 是 C++ 标准数据类型中使用, 比如int, short, long, long long, float, double, long double 以及 char 等
自定义数据类型需要重载运算符。

而 half 是 CUDA 中的数据类型，只能使用 CUDA 提供的函数进行运算。
*/
__device__ __forceinline__ half elu_half(half x){
    return __hgt(x, __float2half(0.f)) ? x : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}

// CUDA 核函数
// -------- FP32 ----------
__global__ void elu_f32_kernel(float* x, float* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] = elu(x[idx]);
    }
}


__global__ void elu_f32x4_kernel(float* x, float* y, int N){
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if(idx < N){
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = elu(reg_x.x);
        reg_y.y = elu(reg_x.y);
        reg_y.z = elu(reg_y.z);
        reg_y.w = elu(reg_y.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

// -------- FP16 ----------
__global__ void elu_f16_kernel(half* x, half* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] = elu_half(x[idx]);
    }
}


__global__ void elu_f16x2_kernel(half* x, half* y, int N){
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if(idx < N){
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = elu_half(reg_x.x);
        reg_y.y = elu_half(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

__global__ void elu_f16x8_kernel(half* x, half* y, int N){
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    half2 reg_x_0 = HALF2(x[idx]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = elu_half(reg_x_0.x);
    reg_y_0.y = elu_half(reg_x_0.y);
    reg_y_1.x = elu_half(reg_x_1.x);
    reg_y_1.y = elu_half(reg_x_1.y);
    reg_y_2.x = elu_half(reg_x_2.x);
    reg_y_2.y = elu_half(reg_x_2.y);
    reg_y_3.x = elu_half(reg_x_3.x);
    reg_y_3.y = elu_half(reg_x_3.y);
    if(idx < N){
        HALF2(y[idx]) = reg_y_0;
    }
    if(idx + 2 < N){
        HALF2(y[idx + 2]) = reg_y_1;
    }
    if(idx + 4 < N){
        HALF2(y[idx + 4]) = reg_y_2;
    }
    if(idx + 6 < N){
        HALF2(y[idx + 6]) = reg_y_3;
    }
}


__global__ void elu_f16x8_pack_kernel(half* x, half* y, int N){
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x) = LDST128BITS(x[idx]);

    #pragma unroll
    for(int i = 0;i<8;i++){
        pack_y[i] = elu_half(pack_x[i]);
    }
    if(idx + 7 < N){
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}


// 定义 CHECK_TORCH_TENSOR_DTYPE 宏
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if((T).options().dtype() != (th_type)){   \
        std::cout<< "Tensor Info: " << (T).options() << std::endl;  \
        throw std::runtime_error("Tensor dtype must be " #th_type);   \
    }      



// 定义 TORCH_BINDING_COMMON_EXTENSION 宏
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func))

// PyTorch 绑定代码
#define TORCH_BINDING_ELU(packed_type, th_type, element_type, n_elements)      \
void elu_##packed_type(torch::Tensor x, torch::Tensor y) {                     \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
  const int ndim = x.dim();                                                  \
  if (ndim != 2) {                                                           \
    int N = 1;                                                             \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                     \
    dim3 block(256 / (n_elements));                                        \
    dim3 grid((N + 256 - 1) / 256);                                        \
    elu_##packed_type##_kernel<<<grid, block>>>(                           \
        reinterpret_cast<element_type*>(x.data_ptr()),                     \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                 \
  } else {                                                                   \
    const int S = x.size(0);                                               \
    const int K = x.size(1);                                               \
    const int N = S * K;                                                   \
    if ((K/(n_elements)) <= 1024) {                                        \
        dim3 block(K/(n_elements));                                        \
        dim3 grid(S);                                                      \
        elu_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type*>(x.data_ptr()),                 \
            reinterpret_cast<element_type*>(y.data_ptr()), N);             \
    } else {                                                               \
        int N = 1;                                                         \
        for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                 \
        dim3 block(256 / (n_elements));                                    \
        dim3 grid((N + 256 - 1) / 256);                                    \
        elu_##packed_type##_kernel<<<grid, block>>>(                       \
        reinterpret_cast<element_type*>(x.data_ptr()),                 \
        reinterpret_cast<element_type*>(y.data_ptr()), N);             \
        }                                                                      \
    }                                                                          \
}

TORCH_BINDING_ELU(f32,        torch::kFloat32,    float,    1)
TORCH_BINDING_ELU(f32x4,      torch::kFloat32,    float,    4)
TORCH_BINDING_ELU(f16,        torch::kHalf,       half,     1)
TORCH_BINDING_ELU(f16x2,      torch::kHalf,       half,     2)
TORCH_BINDING_ELU(f16x8,      torch::kHalf,       half,     8)
TORCH_BINDING_ELU(f16x8_pack, torch::kHalf,       half,     8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elu_f32)
  TORCH_BINDING_COMMON_EXTENSION(elu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8_pack)
}
