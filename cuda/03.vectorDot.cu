// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 
In a parallel computing environment like CUDA, multiple threads may try to read, modify, 
and write back the same memory location simultaneously. 
This can lead to race conditions, where the final result depends on the order of operations, 
which is not predictable in a parallel environment. Atomic operations prevent these race conditions by ensuring 
that the read-modify-write operations are done as a single, indivisible operation. 
This means that once a thread starts an atomic operation, no other thread can start another atomic operation 
on the same memory location until the first one is completed.

For dot products, a · b = a1 * b1 + a2 * b2 + ... + an * bn,

We use atomic addition to aggregate the results of the multiplication of each element of the vectors.
*/

// CUDA kernel function to compute the dot product of two vectors
__global__ void vectorDot(const float *a, const float *b, float *c, int numElements) {
    // Calculate the unique index for the thread
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Check if index is within the range of elements
    if (idx < numElements){
        // Perform the multiplication and add to the result
        atomicAdd(c, a[idx] * b[idx]);
    }
}

// Function to generate a random float
float randomFloat(int randMax = 1000){
    // Return a random float between 0 and 1
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

// Main function
int main(){
    // Seed the random number generator
    srand(time(0));

    // Define the number of elements in the vectors
    int numElements = 300;

    // Calculate the size of the vectors in bytes
    size_t size = numElements * sizeof(float);
    
    // Declare pointers for host and device vectors
    float *hostA, *hostB, *hostC;
    float *deviceA, *deviceB, *deviceC;

    // Allocate memory for host vectors
    hostA = (float*) malloc(size);
    hostB = (float*) malloc(size);
    hostC = (float*) malloc(sizeof(float)); // Only need space for one float for the result

    // Initialize host vectors with random floats
    for (int idx = 0; idx < numElements; idx++){
        hostA[idx] = randomFloat();
        hostB[idx] = randomFloat();
    }

    // Allocate memory for device vectors
    cudaMalloc((void**)&deviceA, size);
    cudaMalloc((void**)&deviceB, size);
    cudaMalloc((void**)&deviceC, sizeof(float)); // Only need space for one float for the result

    // Copy host vectors to device
    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector dot product kernel
    vectorDot<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numElements);

    // Copy the result from device to host
    cudaMemcpy(hostC, deviceC, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result of the dot product
    std::cout << "Dot product: " << *hostC << std::endl;

    // Free the memory allocated for host vectors
    free(hostA);
    free(hostB);
    free(hostC);

    // Free the memory allocated for device vectors
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}