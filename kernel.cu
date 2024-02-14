#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <winsock2.h>
#include <malloc.h>
#include <chrono>
#include <Windows.h>
#include <stdint.h>
#include <iostream>

// Функция для сложения векторов на CPU
void addVectorCPU(long long* a, long long* b, long long* c, long long n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorAdd(long long* a, long long* b, long long* c, long long n)
{
        
    int i = blockIdx.x * blockDim.x + threadIdx.x; // номер блока, кол-во потоков в блоке, индекс тек потока
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void print(long long* c, long long N)
{
    std::cout << "first 50 el\n\n\n";

    for (int i = 0; i < 50; i++)printf("%d ", c[i]);

    std::cout << "\n\n";

    std::cout << "\n\nLast 50 el\n\n";

    for (int i = N - 50; i < N; i++)printf("%d ", c[i]);

    std::cout << "\n\n";
}

int main() 
{
    const long long N = 100000000;  // Длина векторов
    long long* a, * b, * c;       // Входные и выходной векторы на CPU
    long long* d_a, * d_b, * d_c; // Входные и выходной векторы на GPU


    a = (long long*)malloc(N * sizeof(long long));
    b = (long long*)malloc(N * sizeof(long long));
    c = (long long*)malloc(N * sizeof(long long));

    cudaMalloc((void**)&d_a, N * sizeof(long long));
    cudaMalloc((void**)&d_b, N * sizeof(long long));
    cudaMalloc((void**)&d_c, N * sizeof(long long));

    // Заполнение векторов значениями
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(d_a, a, N * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(long long), cudaMemcpyHostToDevice);

    // Запуск ядра функции vectorAdd на GPU и замерка времени выполнения
    
    auto start = std::chrono::high_resolution_clock::now();


    const int block_size = 1024;
    int num_blocks = (N + block_size - 1) / block_size; 
    vectorAdd <<< num_blocks, block_size>> > (d_a, d_b, d_c, N);

    //vectorAdd <<< 1024, 1024 >>> (d_a, d_b, d_c, N);
    cudaDeviceSynchronize();    // Ожидание завершения всех операций на устройстве
    
    
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;

    cudaMemcpy(c, d_c, N * sizeof(long long), cudaMemcpyDeviceToHost);
    
    

    print(c, N);
    printf("\nGPU Time for N=%d: %lf ms\n", N, duration.count());



    //Выполнение операции сложения на CPU и замерка времени
    start = std::chrono::high_resolution_clock::now();


    addVectorCPU(a, b, c, N);


    end = std::chrono::high_resolution_clock::now();

   duration = end - start;

   print(c, N);

    printf("CPU Time for N=%d: %f ms\n", N, duration);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}


//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//
//
//
//
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    static int z;
//    z++;
//    int i = threadIdx.x;
//    c[z] = a[i] + b[i];
//}
//
//
//__global__ void vectorAdd(int* a, int* b, int* c, int length) {
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    if (tid < length) {
//        c[tid] = a[tid] + b[tid];
//    }
//}
//
//void sumCPU(int sum, int size) {
//    for (int i = 0; i < size; i++)sum++;
//}
//
//int main()
//{
//    int sumCPU = 0, sumGPU = 0;
//
//       
//
//
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//
//
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//    
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
