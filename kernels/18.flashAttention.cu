// FlashAttention: A memory-efficient attention mechanism for transformers.
// This implementation demonstrates the core principles of FlashAttention,
// including tiling, shared memory usage, and online softmax calculation.

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <curand_kernel.h> // Include if using random initialization on GPU
#include <limits> // Required for std::numeric_limits

// --- Constants ---
// Define sequence length, head dimension, and block sizes for tiling
const int SEQ_LEN = 256; // Example sequence length, reduced for faster testing
const int HEAD_DIM = 64;  // Example head dimension (d_k)
const int BLOCK_SIZE_N = 64; // Block size for sequence length dimension (K, V) (Bc in paper) - K/V tile height
const int BLOCK_SIZE_M = 64; // Block size for sequence length dimension (Q) (Br in paper) - Q tile height
// const int BLOCK_SIZE_K = 64;  // Block size for head dimension (d_k) - This is effectively HEAD_DIM


// --- FlashAttention Kernel ---
// Assumptions:
// - N (sequence length) is divisible by BLOCK_SIZE_M for simplicity in this version.
// - d_k (HEAD_DIM) is divisible by a reasonable number of threads for loading, or threads load multiple elements.
// - d_v == d_k
// - Each thread block processes one BLOCK_SIZE_M segment of Q.
// - Each thread within the block processes one row of the Q_i tile.
__global__ void flashAttentionForward(
    const float* Q, // Query matrix (N x d_k)
    const float* K, // Key matrix (N x d_k)
    const float* V, // Value matrix (N x d_v)
    float* O,       // Output matrix (N x d_v)
    int N,          // Sequence length
    int d_k,        // Head dimension (Query and Key)
    int d_v         // Head dimension (Value and Output) - assumed d_v = d_k
) {
    extern __shared__ float s_mem[];
    // sQ: BLOCK_SIZE_M * d_k
    // sK: BLOCK_SIZE_N * d_k
    // sV: BLOCK_SIZE_N * d_v
    float* sQ = s_mem;
    float* sK = sQ + BLOCK_SIZE_M * d_k;
    float* sV = sK + BLOCK_SIZE_N * d_k;

    int row_q_start_idx = blockIdx.x * BLOCK_SIZE_M; // Starting row for this block's Q tile
    int tid_in_block = threadIdx.x; // Thread ID within the block (0 to BLOCK_SIZE_M - 1)

    // Each thread is responsible for one row of Q_i and one row of O_i
    // Check if the current thread's row is within bounds of actual N
    if (row_q_start_idx + tid_in_block >= N) {
        return;
    }

    // Load Q_i tile row into registers (or shared memory if d_k is large)
    // For simplicity, assume Q_i_row fits in registers. If not, sQ would be used more directly.
    // Here, we load the entire sQ tile using all threads in the block.
    // Each thread loads d_k / blockDim.x elements of its assigned Q row.
    // Or, more simply, each thread loads one element at a time in a loop.
    for (int k_idx = tid_in_block; k_idx < BLOCK_SIZE_M * d_k; k_idx += blockDim.x) {
        int r = k_idx / d_k; // row in sQ
        int c = k_idx % d_k; // col in sQ
        if (row_q_start_idx + r < N) { // Boundary check for Q rows
             sQ[r * d_k + c] = Q[(row_q_start_idx + r) * d_k + c];
        } else {
             sQ[r * d_k + c] = 0.0f; // Padding
        }
    }
    // If BLOCK_SIZE_M is not equal to blockDim.x, the above sQ loading needs adjustment.
    // Assuming blockDim.x == BLOCK_SIZE_M, each thread tid_in_block loads its row of Q into sQ.
    // This means tid_in_block is the 'r' (row in sQ).
    // So, thread tid_in_block loads Q[ (row_q_start_idx + tid_in_block) * d_k + col_idx ] into sQ[tid_in_block * d_k + col_idx]
    if (blockDim.x == BLOCK_SIZE_M) { // Common case where #threads == Q tile height
        for(int k_loop = 0; k_loop < d_k; ++k_loop) {
            if (row_q_start_idx + tid_in_block < N) {
                 sQ[tid_in_block * d_k + k_loop] = Q[(row_q_start_idx + tid_in_block) * d_k + k_loop];
            } else {
                 sQ[tid_in_block * d_k + k_loop] = 0.0f; // Padding for Q
            }
        }
    }
    __syncthreads(); // Ensure sQ is fully loaded

    // --- Initialize O_i_row, m_i, l_i for the current thread's row ---
    // O_i_row is accumulated in registers of each thread
    float O_i_row[HEAD_DIM]; // HEAD_DIM must be const or use d_v
    for (int i = 0; i < d_v; ++i) O_i_row[i] = 0.0f;

    float m_i = -std::numeric_limits<float>::infinity();
    float l_i = 0.0f;

    // --- Outer loop: Iterate over blocks of K and V (T_c times) ---
    int Tc = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    for (int j_block = 0; j_block < Tc; ++j_block) {
        int row_k_start_idx = j_block * BLOCK_SIZE_N;

        // Load K_j tile into sK and V_j tile into sV
        // Each thread loads a portion of sK and sV.
        // Example: thread tid_in_block helps load rows of sK and sV.
        // This requires careful indexing. A common way: each thread iterates.
        for (int k_idx = tid_in_block; k_idx < BLOCK_SIZE_N * d_k; k_idx += blockDim.x) {
            int r = k_idx / d_k; // row in sK/sV tile
            int c = k_idx % d_k; // col in sK/sV tile
            if (row_k_start_idx + r < N) { // Boundary check for K/V rows
                sK[r * d_k + c] = K[(row_k_start_idx + r) * d_k + c];
                sV[r * d_v + c] = V[(row_k_start_idx + r) * d_v + c]; // Assuming d_k == d_v for now
            } else {
                sK[r * d_k + c] = 0.0f; // Padding
                sV[r * d_v + c] = 0.0f; // Padding
            }
        }
        __syncthreads(); // Ensure sK and sV are loaded

        // --- Compute S_ij = sQ_i * sK_j^T ---
        // S_ij is a (BLOCK_SIZE_M x BLOCK_SIZE_N) matrix.
        // Thread tid_in_block computes one row of S_ij: S_ij[tid_in_block, :]
        // This S_ij_row is stored in registers or a small piece of shared memory if needed.
        float S_ij_row[BLOCK_SIZE_N]; // Store one row of S_ij

        for (int n_idx = 0; n_idx < BLOCK_SIZE_N; ++n_idx) { // Iterate over columns of S_ij (rows of sK_j)
            float score = 0.0f;
            // Dot product of sQ[tid_in_block, :] and sK[n_idx, :]
            for (int dot_k = 0; dot_k < d_k; ++dot_k) {
                score += sQ[tid_in_block * d_k + dot_k] * sK[n_idx * d_k + dot_k];
            }
            S_ij_row[n_idx] = score;
        }

        // --- Online Softmax for the current S_ij_row ---
        float m_i_prev = m_i;
        float l_i_prev = l_i;

        float current_row_max_s = -std::numeric_limits<float>::infinity();
        for (int n_idx = 0; n_idx < BLOCK_SIZE_N; ++n_idx) {
            if (S_ij_row[n_idx] > current_row_max_s) {
                current_row_max_s = S_ij_row[n_idx];
            }
        }

        m_i = max(m_i_prev, current_row_max_s); // New running max

        // Numerically stable P_ij for the current row
        float current_row_sum_p_scaled = 0.0f;
        float P_ij_scaled_row[BLOCK_SIZE_N];

        for (int n_idx = 0; n_idx < BLOCK_SIZE_N; ++n_idx) {
            // If actual K element was padded, its score contributes 0 to softmax
            if (row_k_start_idx + n_idx >= N) {
                 P_ij_scaled_row[n_idx] = 0.0f;
            } else {
                 P_ij_scaled_row[n_idx] = expf(S_ij_row[n_idx] - m_i);
            }
            current_row_sum_p_scaled += P_ij_scaled_row[n_idx];
        }

        l_i = expf(m_i_prev - m_i) * l_i_prev + current_row_sum_p_scaled;

        // --- Update O_i_row ---
        // O_i_row = diag(exp(m_i_prev - m_i) * l_i_prev / l_i) * O_i_row + diag(P_ij_scaled_row / l_i) * sV_j
        float scale_prev_O = expf(m_i_prev - m_i) * (l_i_prev / l_i);
        if (l_i == 0) scale_prev_O = 0; // Avoid division by zero if l_i is zero

        for (int v_col = 0; v_col < d_v; ++v_col) {
            O_i_row[v_col] *= scale_prev_O;
        }

        for (int n_idx = 0; n_idx < BLOCK_SIZE_N; ++n_idx) { // Iterate over rows of sV_j
            float p_val_scaled = P_ij_scaled_row[n_idx] / l_i;
             if (l_i == 0) p_val_scaled = 0; // Avoid division by zero

            for (int v_col = 0; v_col < d_v; ++v_col) { // Iterate over columns of sV_j
                O_i_row[v_col] += p_val_scaled * sV[n_idx * d_v + v_col];
            }
        }
        __syncthreads(); // Ensure all threads in block are done with sK, sV before next K,V load
    } // End of loop over K,V blocks (j_block)

    // --- Write final O_i_row to global memory ---
    // Thread tid_in_block writes its computed O_i_row
    int output_row_global_idx = row_q_start_idx + tid_in_block;
    if (output_row_global_idx < N) { // Check boundary for writing output
        for (int v_col = 0; v_col < d_v; ++v_col) {
            O[output_row_global_idx * d_v + v_col] = O_i_row[v_col];
        }
    }
}

// --- Helper Functions ---

// Function to initialize a matrix on the host
void initializeMatrix(float* matrix, int rows, int cols, bool random = true) {
    for (int i = 0; i < rows * cols; ++i) {
        if (random) {
            matrix[i] = static_cast<float>(rand()) / RAND_MAX;
        } else {
            matrix[i] = 1.0f; // Or some other specific value
        }
    }
}

// Function to print a small part of a matrix
void printMatrix(const float* matrix, int rows, int cols, int print_rows = 5, int print_cols = 5, const std::string& name = "") {
    if (!name.empty()) {
        std::cout << name << ":" << std::endl;
    }
    for (int i = 0; i < std::min(rows, print_rows); ++i) {
        for (int j = 0; j < std::min(cols, print_cols); ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Naive attention implementation (CPU) for verification
void naiveAttentionCPU(
    const float* Q, const float* K, const float* V,
    float* O_cpu,
    int N, int d_k, int d_v
) {
    std::cout << "Running naive attention on CPU for verification..." << std::endl;
    std::vector<float> S(N * N); // Attention score matrix

    // S = Q * K^T
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float score = 0.0f;
            for (int k_idx = 0; k_idx < d_k; ++k_idx) {
                score += Q[i * d_k + k_idx] * K[j * d_k + k_idx];
            }
            S[i * N + j] = score;
        }
    }

    // P = softmax(S)
    std::vector<float> P(N * N); // Softmax probability matrix
    for (int i = 0; i < N; ++i) {
        float row_max = -INFINITY;
        for (int j = 0; j < N; ++j) {
            if (S[i * N + j] > row_max) {
                row_max = S[i * N + j];
            }
        }
        float row_sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            P[i * N + j] = std::exp(S[i * N + j] - row_max);
            row_sum_exp += P[i * N + j];
        }
        for (int j = 0; j < N; ++j) {
            P[i * N + j] /= row_sum_exp;
        }
    }

    // O = P * V
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d_v; ++j) {
            float val = 0.0f;
            for (int k_idx = 0; k_idx < N; ++k_idx) { // k_idx here iterates through sequence length
                val += P[i * N + k_idx] * V[k_idx * d_v + j];
            }
            O_cpu[i * d_v + j] = val;
        }
    }
    std::cout << "Naive attention on CPU finished." << std::endl;
}


// --- Main Function ---
int main() {
    // --- Set matrix dimensions ---
    int N = SEQ_LEN;    // Sequence length
    int d_k = HEAD_DIM; // Dimension of Query and Key vectors
    int d_v = HEAD_DIM; // Dimension of Value and Output vectors (often same as d_k)

    std::cout << "FlashAttention Simulation" << std::endl;
    std::cout << "Sequence Length (N): " << N << std::endl;
    std::cout << "Head Dimension (d_k, d_v): " << d_k << std::endl;

    // --- Allocate memory ---
    // Host memory
    float *h_Q, *h_K, *h_V, *h_O, *h_O_cpu_ref;
    h_Q = new float[N * d_k];
    h_K = new float[N * d_k];
    h_V = new float[N * d_v];
    h_O = new float[N * d_v]; // Output from FlashAttention
    h_O_cpu_ref = new float[N * d_v]; // Output from naive CPU attention

    // Device memory
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, N * d_k * sizeof(float));
    cudaMalloc(&d_K, N * d_k * sizeof(float));
    cudaMalloc(&d_V, N * d_v * sizeof(float));
    cudaMalloc(&d_O, N * d_v * sizeof(float));

    // --- Initialize data ---
    srand(42); // Seed for reproducibility
    initializeMatrix(h_Q, N, d_k);
    initializeMatrix(h_K, N, d_k);
    initializeMatrix(h_V, N, d_v);

    // Copy data from host to device
    cudaMemcpy(d_Q, h_Q, N * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N * d_v * sizeof(float), cudaMemcpyHostToDevice);

    // --- Define kernel launch parameters ---
    // --- Define kernel launch parameters ---
    // Each block processes BLOCK_SIZE_M rows of Q.
    // Each thread in a block processes 1 row of Q. So, BLOCK_SIZE_M threads per block.
    dim3 threadsPerBlock(BLOCK_SIZE_M);
    dim3 numBlocks((N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    // Calculate shared memory size
    // sQ: BLOCK_SIZE_M * d_k
    // sK: BLOCK_SIZE_N * d_k
    // sV: BLOCK_SIZE_N * d_v (assuming d_v = d_k)
    size_t shared_mem_size_bytes = (BLOCK_SIZE_M * d_k + BLOCK_SIZE_N * d_k + BLOCK_SIZE_N * d_v) * sizeof(float);
    std::cout << "Shared memory per block (bytes): " << shared_mem_size_bytes << std::endl;
    // Check if shared memory exceeds device limits (typically 48KB or more)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (shared_mem_size_bytes > prop.sharedMemPerBlock) {
        std::cerr << "ERROR: Requested shared memory (" << shared_mem_size_bytes
                  << " bytes) exceeds device limit (" << prop.sharedMemPerBlock
                  << " bytes per block)." << std::endl;
        // Handle error: clean up and exit
        delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O; delete[] h_O_cpu_ref;
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        return 2;
    }


    std::cout << "Launching FlashAttention kernel with gridDim (" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z
              << ") and blockDim (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << ")" << std::endl;

    // --- Launch FlashAttention kernel ---
    flashAttentionForward<<<numBlocks, threadsPerBlock, shared_mem_size_bytes>>>(
        d_Q, d_K, d_V, d_O, N, d_k, d_v
    );
    cudaDeviceSynchronize(); // Wait for kernel to finish
    std::cout << "FlashAttention kernel finished." << std::endl;

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        // Free memory and exit or handle error
        delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O; delete[] h_O_cpu_ref;
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        return 1;
    }

    // --- Copy output from device to host ---
    cudaMemcpy(h_O, d_O, N * d_v * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Verification (optional) ---
    std::cout << "Running naive CPU attention for verification..." << std::endl;
    naiveAttentionCPU(h_Q, h_K, h_V, h_O_cpu_ref, N, d_k, d_v);

    // Compare h_O and h_O_cpu_ref
    bool results_match = true;
    float tolerance = 1e-2; // Increased tolerance for complex fp ops
    int mismatches_found = 0;
    int max_mismatches_to_print = 10;

    for (int i = 0; i < N * d_v; ++i) {
        if (std::abs(h_O[i] - h_O_cpu_ref[i]) > tolerance) {
            if (mismatches_found < max_mismatches_to_print) {
                // std::cout << "Mismatch at index " << i << " (row " << i/d_v << ", col " << i%d_v
                //           << "): FlashAttention=" << h_O[i] << ", CPU_Ref=" << h_O_cpu_ref[i]
                //           << ", Diff=" << std::abs(h_O[i] - h_O_cpu_ref[i]) << std::endl;
            }
            results_match = false;
            mismatches_found++;
            // break; // Uncomment to stop at first mismatch
        }
    }

    if (results_match) {
        std::cout << "SUCCESS: FlashAttention output matches naive CPU implementation (within tolerance " << tolerance << ")." << std::endl;
    } else {
        std::cout << "FAILURE: FlashAttention output DOES NOT match naive CPU implementation. Mismatches: " << mismatches_found << "/" << N*d_v << std::endl;
    }

    // Print some output for inspection
    // printMatrix(h_Q, N, d_k, 5, 5, "Query (Host)");
    // printMatrix(h_K, N, d_k, 5, 5, "Key (Host)");
    // printMatrix(h_V, N, d_v, 5, 5, "Value (Host)");
    printMatrix(h_O, N, d_v, 5, 5, "Output (FlashAttention)");
    printMatrix(h_O_cpu_ref, N, d_v, 5, 5, "Output (CPU Reference)");


    // --- Free memory ---
    std::cout << "Freeing memory..." << std::endl;
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_O_cpu_ref;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    std::cout << "Finished." << std::endl;
    return 0;
}
