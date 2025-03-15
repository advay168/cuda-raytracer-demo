#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WIDTH 720
#define HEIGHT 480
#define PIXEL_DIM_SPLIT 1
#define PIXEL_SIZE sizeof(v3) * WIDTH *HEIGHT
#define SAMPLES 10
#define MAX_DEPTH 50
#define RAND_COUNT_THREAD 8 * MAX_DEPTH
#define RANDS_COUNT RAND_COUNT_THREAD *SAMPLES *PIXEL_DIM_SPLIT
#define RANDS_SIZE sizeof(float) * RANDS_COUNT
#define MEM_LENGTH WIDTH *HEIGHT * 3
#define SPHERE_COUNT 175
#define PPM_HEADER_FORMAT "P3\n%d %d\n255\n"
#define FRAMES 1

void check(int pos){
  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error at %d: %s\n", pos, cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %d %s\n", pos, cudaGetErrorString(errAsync));
}

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

float rand_float(int *_, float min, float max) {
  float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
  return min + scale * (max - min);       /* [min, max] */
}

typedef struct v3 {
  float x;
  float y;
  float z;
} v3;

__device__ __host__ void v3_add(v3 *lhs, const v3 rhs) {
  lhs->x += rhs.x;
  lhs->y += rhs.y;
  lhs->z += rhs.z;
}

__device__ __host__ void v3_subtract(v3 *lhs, const v3 rhs) {
  lhs->x -= rhs.x;
  lhs->y -= rhs.y;
  lhs->z -= rhs.z;
}

__device__ void v3_multiply(v3 *lhs, const v3 rhs) {
  lhs->x *= rhs.x;
  lhs->y *= rhs.y;
  lhs->z *= rhs.z;
}

__device__ v3 v3_add_to(const v3 a, const v3 b) {
  return {
      a.x + b.x,
      a.y + b.y,
      a.z + b.z,
  };
}

__device__ __host__ v3 v3_subtract_to(const v3 a, const v3 b) {
  return {
      a.x - b.x,
      a.y - b.y,
      a.z - b.z,
  };
}

__device__ __host__ void v3_scale(v3 *from, const float scale) {
  from->x *= scale;
  from->y *= scale;
  from->z *= scale;
}

__device__ __host__ v3 v3_scale_to(const v3 vector, const float scale) {
  return {
      vector.x * scale,
      vector.y * scale,
      vector.z * scale,
  };
}

__device__ __host__ float v3_magnitude(const v3 vector) {
  return fast_sqrtf((vector.x * vector.x) + (vector.y * vector.y) +
                    (vector.z * vector.z));
}

__device__ float v3_dot(const v3 a, const v3 b) {
  return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__host__ v3 v3_cross(const v3 a, const v3 b) {
  v3 cross;
  cross.x = (a.y * b.z) - (a.z * b.y);
  cross.y = (a.z * b.x) - (a.x * b.z);
  cross.z = (a.x * b.y) - (a.y * b.x);
  return cross;
}

__device__ __host__ v3 v3_unit(const v3 vec) {
  v3 out = vec;
  float len = 1.0F / v3_magnitude(vec);
  v3_scale(&out, len);
  return out;
}

typedef struct Ray {
  v3 ori;
  v3 dir;
} Ray;

__device__ __host__ v3 ray_point(const Ray *ray, const float len) {
  v3 p = ray->dir;
  v3_scale(&p, len);
  v3_add(&p, ray->ori);
  return p;
}

struct ColorPick {
  v3 color;
  Ray scatter;
  uint16_t depth;
  bool done;
};

struct DeviceRands {
  const float *nums;
  size_t idx;
};

struct Camera {
  v3 origin;
  v3 lowerLeft;
  v3 horizontal;
  v3 vertical;
  v3 u;
  v3 v;
  v3 w;
  float lensRadius;
};

void Camera_set_view(Camera *cam, const v3 origin, const v3 lookAt, const v3 up,
                     const float fov, const float aspect,
                     const float focalDistance) {
  const float theta = (float)(fov * M_PI / 180.0F);
  const float halfHeight = (theta / 2.0F);
  const float halfWidth = aspect * halfHeight;
  cam->origin = origin;
  cam->w = v3_unit(v3_subtract_to(origin, lookAt));
  cam->u = v3_unit(v3_cross(up, cam->w));
  cam->v = v3_cross(cam->w, cam->u);
  cam->lowerLeft = cam->origin;
  v3_subtract(&cam->lowerLeft, v3_scale_to(cam->u, halfWidth * focalDistance));
  v3_subtract(&cam->lowerLeft, v3_scale_to(cam->v, halfHeight * focalDistance));
  v3_subtract(&cam->lowerLeft, v3_scale_to(cam->w, focalDistance));
  cam->horizontal = cam->u;
  v3_scale(&cam->horizontal, halfWidth * 2.0F * focalDistance);
  cam->vertical = cam->v;
  v3_scale(&cam->vertical, halfHeight * 2.0F * focalDistance);
}

Camera *Camera_new(const v3 origin, const v3 lookAt, const v3 up,
                   const float fov, const float aspect, const float aperture,
                   const float focalDistance) {
  Camera *cam = (Camera *)calloc(1, sizeof(Camera));
  cam->lensRadius = aperture / 2.0F;
  Camera_set_view(cam, origin, lookAt, up, fov, aspect, focalDistance);
  return cam;
}

__device__ v3 local_random_unit_disk(DeviceRands *rng) {
  v3 p = {rng->nums[rng->idx++], rng->nums[rng->idx++], 0.0F};
  v3_scale(&p, 2.0F);
  v3_subtract(&p, {1.0F, 1.0F, 0.0F});
  if (v3_dot(p, p) >= 1.0F)
    v3_scale(&p, 0.5F);
  return p;
}

__device__ Ray Camera_ray(Camera *cam, const float u, const float v,
                          DeviceRands *rng) {
  v3 r = v3_scale_to(local_random_unit_disk(rng), cam->lensRadius);
  v3 offset = v3_scale_to(cam->u, r.x);
  v3_add(&offset, v3_scale_to(cam->v, r.y));
  v3 dir = cam->lowerLeft;
  v3_add(&dir, v3_scale_to(cam->horizontal, u));
  v3_add(&dir, v3_scale_to(cam->vertical, v));
  v3 oo = v3_add_to(cam->origin, offset);
  v3_subtract(&dir, oo);
  return {oo, dir};
}

struct Material;
struct Hit {
  v3 p;    // Point
  v3 nml;  // Normal
  float t; // Time (for like motion blur or w/e)
  const Material *mat;
};

struct Material {
  v3 albedo;
  float fuzz;
  float refractIdx;
  int32_t scatterType;
};

__device__ v3 rnd_unit_sphere_point(DeviceRands *rng) {
  v3 p = {rng->nums[rng->idx++], rng->nums[rng->idx++], rng->nums[rng->idx++]};
  if (v3_magnitude(p) > 0.5F)
    v3_scale(&p, 0.5F);
  return p;
}

__device__ float schlick(const float cosine, const float refractIdx) {
  float r0 = (1 - refractIdx) / (1 + refractIdx);
  r0 = r0 * r0;
  float a = 1 - cosine;
  return r0 + (1.0F - r0) * a * a * a * a * a;
}

__device__ v3 reflect(const v3 v, const v3 n) {
  const float dot = v3_dot(v, n);
  v3 r = v3_scale_to(n, dot * 2.0F);
  return v3_subtract_to(v, r);
}

__device__ bool refract(const v3 v, const v3 n, const float niOverNt,
                        v3 *outRefracted) {
  v3 uv = v3_unit(v);
  const float dt = v3_dot(uv, n);
  const float discriminant = 1.0F - niOverNt * niOverNt * (1.0F - dt * dt);
  if (discriminant > 0.0F) {
    v3 l = v3_scale_to(v3_subtract_to(uv, v3_scale_to(n, dt)), niOverNt);
    v3 r = v3_scale_to(n, fast_sqrtf(discriminant));
    *outRefracted = v3_subtract_to(l, r);
    return true;
  }
  return false;
}

__device__ bool mat_scatter_lambert(const Hit *hit, v3 *outAttenuation,
                                    Ray *outRay, DeviceRands *rng) {
  v3 target = v3_add_to(hit->p, hit->nml);
  v3_add(&target, rnd_unit_sphere_point(rng));
  *outRay = {hit->p, v3_subtract_to(target, hit->p)};
  *outAttenuation = hit->mat->albedo;
  return true;
}

__device__ bool mat_scatter_metal(const Hit *hit, const Ray *ray,
                                  v3 *outAttenuation, Ray *outRay,
                                  DeviceRands *rng) {
  v3 reflected = reflect(v3_unit(ray->dir), hit->nml);
  v3 dir = v3_scale_to(rnd_unit_sphere_point(rng), hit->mat->fuzz);
  v3_add(&dir, reflected);
  *outRay = {hit->p, dir};
  *outAttenuation = hit->mat->albedo;
  return v3_dot(outRay->dir, hit->nml) > 0.0F;
}

__device__ bool mat_scatter_dielectric(const Hit *hit, const Ray *ray,
                                       v3 *outAttenuation, Ray *outRay,
                                       DeviceRands *rng) {
  v3 oNorm;
  float niOverNt;
  float reflectProb;
  float cosine;

  *outAttenuation = {1.0F, 1.0F, 1.0F};
  if (v3_dot(ray->dir, hit->nml) > 0.0F) {
    oNorm = v3_scale_to(hit->nml, -1.0F);
    niOverNt = hit->mat->refractIdx;
    cosine = v3_dot(ray->dir, hit->nml) / v3_magnitude(ray->dir);
    cosine = fast_sqrtf(1.0F - hit->mat->refractIdx * hit->mat->refractIdx *
                                   (1.0F - cosine * cosine));
  } else {
    oNorm = hit->nml;
    niOverNt = 1.0F / hit->mat->refractIdx;
    cosine = -v3_dot(ray->dir, hit->nml) / v3_magnitude(ray->dir);
  }
  v3 refracted;
  v3 reflected = reflect(ray->dir, hit->nml);
  if (refract(ray->dir, oNorm, niOverNt, &refracted))
    reflectProb = schlick(cosine, hit->mat->refractIdx);
  else
    reflectProb = 1.0F;
  if (rng->nums[rng->idx++] < reflectProb)
    *outRay = {hit->p, reflected};
  else
    *outRay = {hit->p, refracted};
  return true;
}

Material mat_lambert(const v3 albedo) {
  return {
      .albedo = albedo,
      .scatterType = 1,
  };
}

Material mat_metal(const v3 albedo, const float fuzz) {
  return {
      .albedo = albedo,
      .fuzz = fuzz,
      .scatterType = 2,
  };
}

Material mat_glass(const float refractIndex) {
  return {
      .refractIdx = refractIndex,
      .scatterType = 3,
  };
}

struct Sphere {
  v3 center;
  float radius;
  Material mat;
};

__device__ bool sphere_hit(const Sphere *sphere, const Ray *ray,
                           const float tMin, float tMax, Hit *outHit) {
  v3 oc = v3_subtract_to(ray->ori, sphere->center);
  float a = v3_dot(ray->dir, ray->dir);
  float b = v3_dot(oc, ray->dir);
  float c = v3_dot(oc, oc) - sphere->radius * sphere->radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0.0F) {
    float t = 0.0F;
    float t0 = (-b - fast_sqrtf(discriminant)) / a;
    float t1 = (-b + fast_sqrtf(discriminant)) / a;
    if (t0 < tMax && t0 > tMin)
      t = t0;
    else if (t1 < tMax && t1 > tMin)
      t = t1;
    else
      return false;
    outHit->t = t;
    outHit->p = ray_point(ray, t);
    outHit->nml = outHit->p;
    v3_subtract(&outHit->nml, sphere->center);
    v3_scale(&outHit->nml, 1.0F / sphere->radius);
    outHit->mat = &sphere->mat;
    return true;
  }
  return false;
}

__device__ void color(Ray *ray, Sphere *spheres, const size_t sphereCount,
                      DeviceRands *rng, ColorPick *pick) {
  Hit hit;
  hit.t = FLT_MAX;
  for (size_t i = 0; i < sphereCount; ++i) {
    Hit check;
    check.t = FLT_MAX;
    sphere_hit(spheres + i, ray, 0.001F, FLT_MAX, &check);
    /*--- Near 0 should be ignored, thus 0.001F ----------------------------*/
    if (check.t < hit.t)
      hit = check;
  }
  if (hit.t < FLT_MAX) {
    Ray scattered;
    v3 attenuation;
    bool valid = false;
    switch (hit.mat->scatterType) {
    case 1:
      valid = mat_scatter_lambert(&hit, &attenuation, &scattered, rng);
      break;
    case 2:
      valid = mat_scatter_metal(&hit, ray, &attenuation, &scattered, rng);
      break;
    case 3:
      valid = mat_scatter_dielectric(&hit, ray, &attenuation, &scattered, rng);
      break;
    }
    if (valid) {
      pick->scatter = scattered;
      v3_multiply(&pick->color, attenuation);
      return;
    } else {
      pick->color = {0, 0, 0};
      pick->done = true;
      return;
    }
  }
  // 1 = blue & 0 = white
  v3 unit = v3_unit(ray->dir);      // -1 < y < 1
  float t = 0.5F * (unit.y + 1.0F); // Scale above to 0 < y < 1
  /*--- ((1.0 - t) * <1.0,1.0,1.0>) + (t * <0.5,0.7,0.1>) ----------------*/
  v3 start = {1, 1, 1};
  v3 end = {0.5F, 0.7F, 1.0F};
  v3_scale(&end, t);
  v3_scale(&start, 1.0F - t);
  v3_add(&start, end);
  v3_multiply(&pick->color, start);
  pick->done = true;
}

void write_image(const uint8_t *buff, const size_t len, const int32_t width,
                 const int32_t height, const int32_t frameIdx) {
  char name[128];
  snprintf(name, 128, "frame-%d.ppm", frameIdx);
  FILE *fp = fopen(name, "w");

  char header[sizeof(PPM_HEADER_FORMAT) + 20];
  snprintf(header, sizeof(header), PPM_HEADER_FORMAT, width, height);
  fwrite(header, strlen(header), 1, fp);

  const size_t colCount = (size_t)width * 3;
  for (size_t i = 0; i < len;) {
    for (size_t c = 0; c < colCount; ++c, ++i) {
      fprintf(fp, "%d ", buff[i]);
    }
    fwrite("\n", 1, 1, fp);
  }

  fclose(fp);
}

Sphere *get_spheres(const size_t count, size_t *outCount) {
  int r = 1;
  int *rnd = &r;

  Sphere *spheres = (Sphere *)malloc(sizeof(Sphere) * (count + 1));
  size_t i = 0;

  spheres[i++] = {
      {0.0F, -1000.0F, 0.0F}, 1000.0F, mat_lambert({0.5F, 0.5F, 0.5F})};
  const v3 m = {4.0F, 0.2F, 0.0F};
  for (int32_t a = -5; a < 5; ++a) {
    for (int32_t b = -5; b < 5; ++b) {
      v3 center = {a + 0.9F * rand_float(rnd, 0.0F, 1.0F), 0.2F,
                   b + 0.9F * rand_float(rnd, 0.0F, 1.0F)};
      if (v3_magnitude(v3_subtract_to(center, m)) > 0.9F) {
        float matChoice = rand_float(rnd, 0.0F, 1.0F);
        if (matChoice < 0.8F)
          spheres[i++] = {center, 0.2F,
                          mat_lambert({rand_float(rnd, 0.0F, 1.0F),
                                       rand_float(rnd, 0.0F, 1.0F),
                                       rand_float(rnd, 0.0F, 1.0F)})};
        else if (matChoice < 0.95F)
          spheres[i++] = {center, 0.2F,
                          mat_metal({(1 + rand_float(rnd, 0.0F, 1.0F)) * 0.5F,
                                     (1 + rand_float(rnd, 0.0F, 1.0F)) * 0.5F,
                                     (1 + rand_float(rnd, 0.0F, 1.0F)) * 0.5F},
                                    rand_float(rnd, 0.0F, 1.0F) * 0.5F)};
        else
          spheres[i++] = {center, 0.2F, mat_glass(1.5F)};
      }

      if (i == count - 3)
        break;
    }

    if (i == count - 3)
      break;
  }
  spheres[i++] = {{0.0F, 1.0F, 0.0F}, 1.0F, mat_glass(1.5F)};
  spheres[i++] = {{-4.0F, 1.0F, 0.0F}, 1.0F, mat_lambert({0.4F, 0.2F, 0.1F})};
  spheres[i++] = {
      {4.0F, 1.0F, 0.0F}, 1.0F, mat_metal({0.7F, 0.6F, 0.5F}, 0.0F)};
  *outCount = i;
  return spheres;
}

__global__ void dummy(Camera *cam, Sphere *spheres, size_t sphereCount, v3 *rgb,
                      const float *dRands, Sphere *cuda_Spheres) {
  // __shared__ v3 samples[SAMPLES];
  // int threadPositionInBlock = threadIdx.x + blockDim.x * threadIdx.y +
  //                             blockDim.x * blockDim.y * threadIdx.z;
  int h = (blockIdx.x * blockDim.y) + threadIdx.y;
  int idx = (blockIdx.y * gridDim.x * blockDim.y) + h;
  rgb[idx] = {0, 0, 0};

  ColorPick pick;
  DeviceRands rng;
  rng.idx = 0;
  int rndOffset = threadIdx.x * RAND_COUNT_THREAD;
  rng.nums = dRands + rndOffset;
  const float u = ((float)h + rng.nums[rng.idx++]) / WIDTH;
  const float v = ((float)HEIGHT - blockIdx.y + rng.nums[rng.idx++]) / HEIGHT;
  pick.scatter = Camera_ray(cam, u, v, &rng);
  pick.done = false;
  pick.depth = 0;
  pick.color = {1, 1, 1};
  while (!pick.done && pick.depth < MAX_DEPTH) {
    color(&pick.scatter, cuda_Spheres, sphereCount, &rng, &pick);
    pick.depth++;
  }

  // samples[threadPositionInBlock] = pick.color;
  // Block for all threads and sum up shared values
  // __syncthreads();
  // if (threadPositionInBlock == 0) {
  //   v3 c = {0};
  //   for (int i = 0; i < SAMPLES; ++i)
  //     v3_add(&c, samples[i]);
  //   rgb[idx] = c;
  // }

  rgb[idx] = v3_add_to(rgb[idx], pick.color);

  // v3 c = pick.color;
  // atomicAdd(&rgb[idx].x, c.x);
  // atomicAdd(&rgb[idx].y, c.y);
  // atomicAdd(&rgb[idx].z, c.z);
}

void render_frame(Sphere *spheres, size_t sphereCount) {
  float scale = 1.0F / FRAMES;
  float cx = 8.0F;
  float cz = 10.0F;
  const float circle = (float)(M_PI * 2.0F);
  float angleDelta = circle * scale;
  float angle = 0.0F;

  Sphere *dSpheres;
  cudaMalloc(&dSpheres, sizeof(Sphere) * sphereCount);
  cudaMemcpy(dSpheres, spheres, sizeof(Sphere) * sphereCount,
             cudaMemcpyHostToDevice);

  Camera *dCamera;
  cudaMalloc(&dCamera, sizeof(Camera));

  v3 *dCp;
  cudaMalloc(&dCp, PIXEL_SIZE);
  v3 *cp;
  cudaMallocHost(&cp, PIXEL_SIZE);

  float *dRands;
  cudaMalloc(&dRands, RANDS_SIZE);

  dim3 block;
  block.x = SAMPLES;
  block.y = PIXEL_DIM_SPLIT;
  block.z = 1;

  dim3 grid;
  grid.x = WIDTH / PIXEL_DIM_SPLIT;
  grid.y = HEIGHT;
  grid.z = 1;

  // const size_t sharedMemSize = sizeof(v3) * SAMPLES;

  const float fov = 20.0F;
  const float aspect = (float)WIDTH / (float)HEIGHT;
  const v3 lookAt = {0, 0, 0};
  const float aperture = 0.1F;
  const float focalDist = 10.0F;
  uint8_t *data = (uint8_t *)malloc(MEM_LENGTH);
  Camera *cam = Camera_new({cx, 2.0F, cz}, lookAt, {0, 1, 0}, fov, aspect,
                           aperture, focalDist);

  Sphere *cuda_Spheres;
  cudaMalloc(&cuda_Spheres, sizeof(Sphere) * 500);
  cudaMemcpy(cuda_Spheres, spheres, sphereCount * sizeof(Sphere),
             cudaMemcpyHostToDevice);

  float *hRands = (float *)malloc(RANDS_SIZE);
  for (int i = 0; i < RANDS_COUNT; i++) {
    hRands[i] = rand_float(NULL, 0, 1);
  }
  cudaMemcpy(dRands, hRands, RANDS_SIZE, cudaMemcpyHostToDevice);

  for (int32_t f = 0; f < FRAMES; ++f) {
    cam->origin.x = cx * cosf(angle) - cz * sinf(angle);
    cam->origin.z = cz * cosf(angle) + cx * sinf(angle);
    Camera_set_view(cam, cam->origin, lookAt, {0, 1, 0}, fov, aspect,
                    focalDist);
    angle += angleDelta;

    cudaMemcpy(dCamera, cam, sizeof(Camera), cudaMemcpyHostToDevice);
    check(0);
    dummy<<<grid, block>>>(dCamera, dSpheres, sphereCount, dCp, dRands,
                           cuda_Spheres);
    check(1);
    cudaDeviceSynchronize();
    check(2);

    cudaMemcpy(cp, dCp, PIXEL_SIZE, cudaMemcpyDeviceToHost);

    size_t writeIdx = 0;
    for (int32_t i = 0; i < WIDTH * HEIGHT; ++i) {
      v3 c = cp[i];
      v3_scale(&c, 1.0F / SAMPLES);
      c = {fast_sqrtf(c.x), fast_sqrtf(c.y), fast_sqrtf(c.z)};

      data[writeIdx++] = c.x * 255.99F;
      data[writeIdx++] = c.y * 255.99F;
      data[writeIdx++] = c.z * 255.99F;
    }
    write_image(data, MEM_LENGTH, WIDTH, HEIGHT, f);
  }
  cudaFree(dSpheres);
  cudaFree(dCamera);
  cudaFree(dCp);
  cudaFree(dRands);
  cudaFreeHost(cp);
  free(cam);
}

int main(void) {
  size_t sphereCount = 0;
  Sphere *spheres = get_spheres(SPHERE_COUNT, &sphereCount);
  render_frame(spheres, sphereCount);

  check(3);
  return 0;
}

