#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

#define FULLMASK 0xffffffff

template <typename T>
__device__ T BlockReduce(T val)
{
    const int tid = threadIdx.x;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;

    for (size_t offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULLMASK, val, offset);
    }

    
    __shared__ T warp_shared[32];
    if (laneID == 0)
    {
        warp_shared[warpID] = val;
    }
    __syncthreads();

    if (warpID == 0)
    {

        if (laneID < blockDim.x / warpSize) val = warp_shared[laneID];
        else val = 0;

        for (size_t offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(FULLMASK, val, offset);
        }
    }
    return val;
}


template <typename T>
__global__ void trace_kernel(T* d_input, T* d_output, const size_t N)
{
    const size_t tid = threadIdx.x;
    const size_t bid = blockIdx.x;
    const size_t global_id = tid + bid * blockDim.x;

    T sum = (T)0;
    for (size_t i =  global_id; i < N; i += blockDim.x * gridDim.x)
    {
        sum += d_input[i];
    }

    sum = BlockReduce<T>(sum);

    if (tid == 0)
        d_output[bid] = sum;

}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // 1. 正常的对角线提取
    const size_t n_diag = (rows < cols) ? rows : cols;
    std::vector<T> temp;
    temp.reserve(n_diag);
    for (size_t i = 0; i < n_diag; i++) {
        temp.push_back(h_input[i * cols + i]);
    }

    const size_t N = temp.size();
    const size_t num_bytes = sizeof(T) * N;

    // 2. 准备设备内存
    T *d_input, *d_output;
    cudaMalloc((void**)&d_input, num_bytes);
    
    // 获取硬件属性
    int deviceID;
    cudaGetDevice(&deviceID);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);
    size_t threads = prop.maxThreadsPerBlock;
    size_t blocks = (N + threads - 1) / threads;

    // 输出只需要存下每个 block 的结果
    cudaMalloc((void**)&d_output, blocks * sizeof(T));
    cudaMemcpy(d_input, temp.data(), num_bytes, cudaMemcpyHostToDevice);


    trace_kernel<T><<<blocks, threads>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    std::vector<T> h_output(blocks);
    cudaMemcpy(h_output.data(), d_output, blocks * sizeof(T), cudaMemcpyDeviceToHost);

    T res = (T)0;
    for (const auto& val : h_output) {
        res += val;
    }

    // 释放资源
    cudaFree(d_input);
    cudaFree(d_output);

    return res;
}
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
