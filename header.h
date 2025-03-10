#pragma once

#include <stdint.h>
struct vec3 {
  float x, y, z;
};

struct sphere {
  vec3 center;
  float radius;
  uint32_t colour;
};

void init(int width, int height);
void doRender(uint32_t *framebuffer, int width, int height, sphere *s,
              int count);
