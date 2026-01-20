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
#define BLOCKSIZE 1024
#define WARPSIZE 32
#define FULLMASK 0xffffffff

template <typename T>
__device__  T BlockReduce(T val)
{
    const int tid = threadIdx.x;
    const int warpID = tid / WARPSIZE;
    const int laneID = tid % WARPSIZE;

    for (size_t offset = WARPSIZE / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULLMASK, val, offset);
    }

    __shared__ T warp_shared[BLOCKSIZE / WARPSIZE];
    if (laneID == 0)
    {
        warp_shared[warpID] = val;
    }
    __syncthreads();

    if (warpID == 0)
    {
        val = warp_shared[laneID];
        for (size_t offset = WARPSIZE / 2; offset > 0; offset >>= 1)
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

    T sum = 0.f;
    for (size_t i =  global_id; i < N; i += blockDim.x * gridDim.x)
    {
        sum += d_input[i];
    }

    sum = BlockReduce(sum);

    if (tid == 0)
        d_output[bid] = sum;

}


template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {

    if (rows > cols) rows = cols;
    else cols = rows;

    std::vector<T> temp;
    for (int i = 0; i < rows; i++)
    {
        temp.push_back(h_input[i * cols + i]);
    }

    const size_t num_bytes = sizeof(T) * temp.size();
    const int N = temp.size();

    T* d_intput,* d_output;
    cudaMalloc((void**)&d_intput, num_bytes);
    cudaMalloc((void**)&d_output, num_bytes);
    cudaMemcpy(d_intput, temp.data(), num_bytes, cudaMemcpyHostToDevice);
    
    size_t threads = BLOCKSIZE;
    size_t n = N;
    size_t blocks = (n + threads - 1) / threads;
    T* d_in = d_intput;
    T* d_out = d_output;

    while (blocks > 1)
    {
        trace_kernel<T><<<blocks, threads>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        n = blocks;
        blocks = (n + threads - 1) / threads;
        std::swap(d_in, d_out);
    }
    trace_kernel<T><<<1, threads>>>(d_in, d_out, n);
    T res = 0;
    cudaMemcpy(&res, d_out, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_intput);


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
