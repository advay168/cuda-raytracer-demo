// Credit to https://github.com/anominos and https://github.com/AdUhTkJm

#include <cuda_runtime.h>
#include <stdio.h>

#include "header.h"

__device__ vec3 operator-(const vec3 &a, const vec3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ float operator*(const vec3 &a, const vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ bool intersect(const sphere &s, const vec3 &orig, const vec3 &dir) {
  vec3 oc = orig - s.center;
  float c = oc * oc - s.radius * s.radius;
  float b = 2.0f * (oc * dir);
  float discriminant = b * b / (dir * dir) - 4 * c;
  if (discriminant < 0)
    return false;

  return discriminant < b * b;
}

// here if needed
__host__ __device__ float fast_sqrtf(float x) {
  unsigned i = *(unsigned *)&x;
  i = (i + 0x3f76cf62) >> 1;
  float y = *(float *)&i;
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  return y;
}

__global__ void render(uint32_t *framebuffer, int width, int height,
                       sphere *spheres, int count) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height)
    return;

  int index = y * width + x;
  float u = (2.0f * x / width - 1.0f) * 2.0f;
  float v = (2.0f * y / height - 1.0f) * 2.0f;
  vec3 ray_origin = {0, 0, 0};
  vec3 ray_direction = {u, v, -1};

  framebuffer[index] = 0xFF000000;
  for (int i = 0; i < count; i++) {
    if (intersect(spheres[i], ray_origin, ray_direction)) {
      framebuffer[index] = spheres[i].colour;
    }
  }
}

uint32_t *d_framebuffer;
sphere *d_sphere = nullptr;
int d_count = 0;

void init(int width, int height) {
  cudaFree(d_framebuffer);
  cudaMalloc(&d_framebuffer, width * height * sizeof(uint32_t));
}

void doRender(uint32_t *framebuffer, int width, int height, sphere *spheres, int count) {
  if (count > d_count) {
    cudaFree(d_sphere);
    d_count = count;
    cudaMalloc(&d_sphere, sizeof(sphere) * d_count);
  }
  cudaMemcpy(d_sphere, spheres, count * sizeof(sphere), cudaMemcpyHostToDevice);

  dim3 blocks(32, 32);
  dim3 threads(width / blocks.x, height / blocks.y);
  // printf("threads(%d, %d, %d)\n", threads.x, threads.y, threads.z);
  // printf("blocks(%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
  render<<<threads, blocks>>>(d_framebuffer, width, height, d_sphere, count);

  cudaMemcpy(framebuffer, d_framebuffer, width * height * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}
