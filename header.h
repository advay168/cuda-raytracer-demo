#pragma once

#include <stdint.h>
struct vec3 {
  float x, y, z;
};

struct Colour {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
};

struct sphere {
  vec3 center;
  float radius;
  Colour colour;
};

void init(int width, int height);
void doRender(Colour *framebuffer, int width, int height, sphere *s,
              int count);
