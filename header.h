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

void init(int w, int h);
void doRender(uint32_t *framebuffer, sphere *s, int count);
