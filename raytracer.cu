#include <cuda_runtime.h>
#include <stdio.h>

#include "header.h"

__host__ __device__ float fast_sqrtf(float x) {
  unsigned i = *(unsigned *)&x;
  i = (i + 0x3f76cf62) >> 1;
  float y = *(float *)&i;
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  return y;
}

__host__ __device__ float Q_rsqrt(float number) {
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y = number;
  i = *(long *)&y;           // evil floating point bit level hacking
  i = 0x5f3759df - (i >> 1); // what the fuck?
  y = *(float *)&i;
  y = y * (threehalfs - (x2 * y * y)); // 1st iteration
  //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can
  // be removed

  return y;
}

__device__ Colour scale(Colour c, float x) {
  return {
    (uint8_t)(c.r * x),
    (uint8_t)(c.g * x),
    (uint8_t)(c.b * x),
    (uint8_t)(c.a),
  };
}

__device__ Colour add(Colour c, Colour x) {
  return {
    (uint8_t)(c.r * x.r),
    (uint8_t)(c.g * x.g),
    (uint8_t)(c.b * x.b),
    (uint8_t)(c.a),
  };
}

__device__ vec3 operator+(const vec3 &a, const vec3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ vec3 operator-(const vec3 &a, const vec3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ float operator*(const vec3 &a, const vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ vec3 operator*(const vec3 &a, const int b) {
  return {a.x * b, a.y * b, a.z * b};
}

__device__ vec3 operator/(const vec3 &a, const int b) {
  return {a.x / b, a.y / b, a.z / b};
}

__device__ float magnitude(const vec3 &a) { return fast_sqrtf(a * a); }
__device__ vec3 normalise(const vec3 &a) { return a / magnitude(a); }

__device__ bool intersect(const sphere &s, const vec3 &O, const vec3 &D, float t,
                          vec3 &hit_position, vec3 &hit_normal) {
  vec3 oc = O - s.center;
  float a = D * D;
  float b = 2.0f * (oc * D);
  float c = oc * oc - s.radius * s.radius;
  float discriminant = b * b - 4 * a * c;

  if (discriminant < 0)
    return false;

  float sqrt_disc = fast_sqrtf(discriminant);
  t = (-b - sqrt_disc) / (2.0f * a);

  if (t < 0)
    return false;

  hit_position = O + D * t;
  hit_normal = normalise(hit_position - s.center);
  return true;
}

__device__ float max_2(float a, float b) { return (a < b) ? b : a; }

__global__ void render(Colour *framebuffer, int width, int height,
                       sphere *spheres, int count) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height)
    return;

  int index = y * width + x;
  float u = (2.0f * x / width - 1.0f) * 2.0f;
  float v = (2.0f * y / height - 1.0f) * 2.0f;
  vec3 ray_origin = {0, 0, 0};
  vec3 ray_direction = {u, v, 1};
  ray_direction = normalise(ray_direction);

  framebuffer[index] = Colour{0, 0,0,255}; // Default black

  float t_min = 1e9;
  vec3 hit_position;
  vec3 hit_normal;
  int hit_index = -1;

  for (int i = 0; i < count; i++) {
    float t;
    vec3 position;
    vec3 normal;
    if (intersect(spheres[i], ray_origin, ray_direction, t, position, normal) &&
        t < t_min) {
      t_min = t;
      hit_position = position;
      hit_normal = normal;
      hit_index = i;
    }
  }

  if (hit_index != -1) {
    vec3 light_pos = {0, 0, -1};
    vec3 light_dir = normalise(light_pos - hit_position);
    float intensity = 10 / (M_PI * 4 * (light_dir * light_dir)) ;
    intensity *= max_2(hit_normal * light_dir, 0);
    Colour colour = spheres[hit_index].colour;

    framebuffer[index] = add(Colour{30, 30, 30, 255}, scale(colour, intensity));
  }
}

Colour *d_framebuffer;
sphere *d_sphere = nullptr;
int d_count = 0;

void init(int width, int height) {
  cudaFree(d_framebuffer);
  cudaMalloc(&d_framebuffer, width * height * sizeof(uint32_t));
}

void doRender(Colour *framebuffer, int width, int height, sphere *spheres,
              int count) {
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
