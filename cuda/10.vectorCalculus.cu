#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

/*

CUDA kernels that calculates the following in paralle on GPU for a batch of inputs:

- The derivative of sin - sin'(x) = (sin(x + h) - sin(x - h)) / 2h
- The integral of sin - ∫sin(x)dx = h * (sin(x) + sin(x - h)) / 2
- The linearization of sin - sin(x) ≈ sin(x0) + sin'(x0) * (x - x0)
- Finding the zero of sin Newton-Raphson method for sin - x = x - (sin(x) - x) / (cos(x) - 1)

The zero of a function is any value within the function's domain that, when input into the function, 
results in an output of zero. In other words, it's the x-value (or input value) 
where the function's graph intersects the x-axis.

N/B: The Newton-Raphson only calculates the first 10 iteration results for demonstrative purposes.
When we take the Newton-Raphson of the derivative of the function, we are finding the zero of the derivative.
That is the point where the derivative (chhange of the function is at a maximum or minimum), aka optimisation.

This is the bedrock of optimization. You can run on even higher order derivatives.

If you use the calculate derivative kernel to calculate the derivative of a function, you can use again  to 
calculate the derivative of its outputs and so on.
*/

__global__ void calculateDerivative(float* input, float* output, float h, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > 0 && idx < numElements - 1) {
        output[idx] = (input[idx + 1] - input[idx - 1]) / (2 * h);
    }
}

__global__ void calculateIntegral(float* input, float* output, float h, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > 0 && idx < numElements - 1) {
        output[idx] = h * (input[idx - 1] + input[idx]) / 2;
    }
}

__global__ void calculateLinearization(float* input, float* derivative, float* output, float h, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > 0 && idx < numElements - 1) {
        float x0 = idx * h;
        output[idx] = input[idx] + derivative[idx] * (x0 - idx * h);
    }
}

__global__ void calculateNewtonRaphson(float* input, float* output, float h, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > 0 && idx < numElements - 1) {
        float x0 = input[idx];
        for (int i = 0; i < 10; i++) { // Perform 10 iterations
            x0 = x0 - (sinf(x0) - x0) / (cosf(x0) - 1);
        }
        output[idx] = x0;
    }
}

int main() {
    int numElements = 1000;
    float h = 0.01f;
    size_t size = numElements * sizeof(float);

    float* hostInput = (float*)malloc(size);
    float* hostOutputDerivative = (float*)malloc(size);
    float* hostOutputIntegral = (float*)malloc(size);
    float* hostOutputLinearization = (float*)malloc(size);
    float* hostOutputNewtonRaphson = (float*)malloc(size);

    for (int i = 0; i < numElements; i++) {
        hostInput[i] = sinf(i * h);
    }

    float* deviceInput;
    float* deviceOutputDerivative;
    float* deviceOutputIntegral;
    float* deviceOutputLinearization;
    float* deviceOutputNewtonRaphson;

    cudaMalloc((void**)&deviceInput, size);
    cudaMalloc((void**)&deviceOutputDerivative, size);
    cudaMalloc((void**)&deviceOutputIntegral, size);
    cudaMalloc((void**)&deviceOutputLinearization, size);
    cudaMalloc((void**)&deviceOutputNewtonRaphson, size);

    cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    calculateDerivative<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutputDerivative, h, numElements);
    calculateIntegral<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutputIntegral, h, numElements);
    calculateLinearization<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutputDerivative, deviceOutputLinearization, h, numElements);
    calculateNewtonRaphson<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutputNewtonRaphson, h, numElements);

    cudaMemcpy(hostOutputDerivative, deviceOutputDerivative, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostOutputIntegral, deviceOutputIntegral, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostOutputLinearization, deviceOutputLinearization, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostOutputNewtonRaphson, deviceOutputNewtonRaphson, size, cudaMemcpyDeviceToHost);

    // Print some values for verification
    for (int i = 0; i < 10; i++) {
        std::cout << "Sine Output: " << hostInput[i] << std::endl;
        std::cout << "Derivative: " << hostOutputDerivative[i] << std::endl;
        std::cout << "Integral: " << hostOutputIntegral[i] << std::endl;
        std::cout << "Linearization: " << hostOutputLinearization[i] << std::endl;
        std::cout << "Newton-Raphson: " << hostOutputNewtonRaphson[i] << "\n" << std::endl;

    }

    free(hostInput);
    free(hostOutputDerivative);
    free(hostOutputIntegral);
    free(hostOutputLinearization);

    cudaFree(deviceInput);
    cudaFree(deviceOutputDerivative);
    cudaFree(deviceOutputIntegral);
    cudaFree(deviceOutputLinearization);

    return 0;
}