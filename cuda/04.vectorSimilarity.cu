// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*

Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space 
that measures the cosine of the angle between them. Its formula is given by: 
cos(θ) = (A . B) / (||A|| * ||B||)
where A and B are vectors and ||A|| and ||B|| are the magnitudes of A and B respectively.

This CUDA implementation shows how to use multiple kernels.

*/

__global__ void dotProduct(const float *a, const float *b, float *c, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(c, a[idx] * b[idx]);
    }
}

__global__ void vectorMagnitude(const float *vector, float *magnitude, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(magnitude, vector[idx] * vector[idx]);
    }
}

float randomFloat(int randMax = 1000){
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

int main(){
    srand(time(0));
    int numElements = 300;
    size_t size = numElements * sizeof(float);
    float *hostVectorA, *hostVectorB, *hostDotProduct, *hostMagnitudeA, *hostMagnitudeB;
    float *deviceVectorA, *deviceVectorB, *deviceDotProduct, *deviceMagnitudeA, *deviceMagnitudeB;

    hostVectorA = (float*) malloc(size);
    hostVectorB = (float*) malloc(size);
    hostDotProduct = (float*) malloc(sizeof(float));
    hostMagnitudeA = (float*) malloc(sizeof(float));
    hostMagnitudeB = (float*) malloc(sizeof(float));

    for (int idx = 0; idx < numElements; idx++){
        hostVectorA[idx] = randomFloat();
        hostVectorB[idx] = randomFloat();
    }

    cudaMalloc((void**)&deviceVectorA, size);
    cudaMalloc((void**)&deviceVectorB, size);
    cudaMalloc((void**)&deviceDotProduct, sizeof(float));
    cudaMalloc((void**)&deviceMagnitudeA, sizeof(float));
    cudaMalloc((void**)&deviceMagnitudeB, sizeof(float));

    cudaMemcpy(deviceVectorA, hostVectorA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorB, hostVectorB, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    dotProduct<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceVectorB, deviceDotProduct, numElements);
    vectorMagnitude<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceMagnitudeA, numElements);
    vectorMagnitude<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorB, deviceMagnitudeB, numElements);

    cudaMemcpy(hostDotProduct, deviceDotProduct, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostMagnitudeA, deviceMagnitudeA, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostMagnitudeB, deviceMagnitudeB, sizeof(float), cudaMemcpyDeviceToHost);

    *hostMagnitudeA = sqrt(*hostMagnitudeA);
    *hostMagnitudeB = sqrt(*hostMagnitudeB);

    float cosineSimilarity = *hostDotProduct / (*hostMagnitudeA * *hostMagnitudeB);

    std::cout << "Cosine Similarity: " << cosineSimilarity << std::endl;

    free(hostVectorA);
    free(hostVectorB);
    free(hostDotProduct);
    free(hostMagnitudeA);
    free(hostMagnitudeB);

    cudaFree(deviceVectorA);
    cudaFree(deviceVectorB);
    cudaFree(deviceDotProduct);
    cudaFree(deviceMagnitudeA);
    cudaFree(deviceMagnitudeB);
}