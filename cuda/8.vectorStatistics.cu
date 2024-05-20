// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 

CUDA kernels that calculates:
- The mean of a vector
- The standard deviation of a vector
- The normalisation of the vector - (xi - μ) / σ
- The standard error of the vector - SE = σ / √n
- The skewness of the vector - (1/n) * Σ[(xi - μ) / σ]^3
- The kurtosis of the vector - (1/n) * Σ[(xi - μ) / σ]^4
- Variance (s²) = (1 / (n - 1)) * Σ(xi - x̄)²
- Covariance = (1 / (n - 1)) * Σ(xi - x̄)(yi - ȳ)

*/

__global__ void vectorMean(const float *vector, float *mean, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(mean, vector[idx]);
    }
}

__global__ void vectorStd(const float *vector, float *std, float *mean, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(std, pow(vector[idx] - *mean, 2));
    }
}

__global__ void vectorNormalization(float *vector, float mean, float std, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        vector[idx] = (vector[idx] - mean) / std;
    }
}

__global__ void vectorSkewnessKurtosis(const float *vector, float *skewness, float *kurtosis, float mean, float std, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        float standardized = (vector[idx] - mean) / std;
        atomicAdd(skewness, pow(standardized, 3));
        atomicAdd(kurtosis, pow(standardized, 4));
    }
}

__global__ void vectorVariance(const float *vector, float *variance, float mean, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(variance, pow(vector[idx] - mean, 2));
    }
}

__global__ void vectorCovariance(const float *vectorX, const float *vectorY, float *covariance, float meanX, float meanY, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(covariance, (vectorX[idx] - meanX) * (vectorY[idx] - meanY));
    }
}

float randomFloat(int randMax = 1000){
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

int main(){
    srand(time(0));
    int numElements = 300;
    size_t size = numElements * sizeof(float);
    float *hostVector, *deviceVector;
    float *hostNormalized, *deviceNormalized;
    float *hostMean, *deviceMean;
    float *hostStd, *deviceStd;
    float *hostSkewness, *deviceSkewness;
    float *hostKurtosis, *deviceKurtosis;
    float *hostVariance, *deviceVariance;
    float *hostCovariance, *deviceCovariance;

    hostVector = (float*) malloc(size);
    hostNormalized = (float*) malloc(size);
    hostMean = (float*) malloc(sizeof(float));
    hostStd = (float*) malloc(sizeof(float));
    hostSkewness = (float*) malloc(sizeof(float));
    hostKurtosis = (float*) malloc(sizeof(float));
    hostVariance = (float*) malloc(sizeof(float));
    hostCovariance = (float*) malloc(sizeof(float));

    // Initialise a random vector of distribution
    for (int idx = 0; idx < numElements; idx++){
        hostVector[idx] = randomFloat();
    }

    // Allocate device vector and variables
    cudaMalloc((void**)&deviceVector, size);
    cudaMalloc((void**)&deviceNormalized, size);
    cudaMalloc((void**)&deviceMean, sizeof(float));
    cudaMalloc((void**)&deviceStd, sizeof(float));
    cudaMalloc((void**)&deviceSkewness, sizeof(float));
    cudaMalloc((void**)&deviceKurtosis, sizeof(float));
    cudaMalloc((void**)&deviceVariance, sizeof(float));
    cudaMalloc((void**)&deviceCovariance, sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(deviceVector, hostVector, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector mean kernel
    vectorMean<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceMean, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostMean, deviceMean, sizeof(float), cudaMemcpyDeviceToHost);
    *hostMean /= numElements;

    // Launch the vector std kernel
    vectorStd<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceStd, deviceMean, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostStd, deviceStd, sizeof(float), cudaMemcpyDeviceToHost);
    *hostStd = sqrt(*hostStd / numElements);

    // Launch the vector normalization kernel
    vectorNormalization<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, *hostMean, *hostStd, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostNormalized, deviceVector, size, cudaMemcpyDeviceToHost);

    // Launch the vector skewness and kurtosis kernel
    vectorSkewnessKurtosis<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceSkewness, deviceKurtosis, *hostMean, *hostStd, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostSkewness, deviceSkewness, sizeof(float), cudaMemcpyDeviceToHost);

    // Launch the vector variance kernel
    vectorVariance<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceVariance, *hostMean, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostVariance, deviceVariance, sizeof(float), cudaMemcpyDeviceToHost);

    // Launch the vector covariance kernel
    vectorCovariance<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceNormalized, deviceCovariance, *hostMean, *hostMean, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostCovariance, deviceCovariance, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Mean: " << *hostMean << std::endl;
    std::cout << "Standard Deviation: " << *hostStd << std::endl;
    std::cout << "Skewness: " << *hostSkewness << std::endl;
    std::cout << "Kurtosis: " << *hostKurtosis << std::endl;
    std::cout << "Variance: " << *hostVariance << std::endl;
    std::cout << "Covariance: " << *hostCovariance << std::endl;

    // Free the memory allocated for host vectors
    free(hostVector);
    free(hostNormalized);
    free(hostMean);
    free(hostStd);
    free(hostSkewness);
    free(hostKurtosis);
    free(hostVariance);
    free(hostCovariance);

    // Free the memory allocated for device vectors
    cudaFree(deviceVector);
    cudaFree(deviceNormalized);
    cudaFree(deviceMean);
    cudaFree(deviceStd);
    cudaFree(deviceSkewness);
    cudaFree(deviceKurtosis);
    cudaFree(deviceVariance);
    cudaFree(deviceCovariance);

}