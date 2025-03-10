#include "header.h"

#include <cmath>
#include <cstdlib>

#include <array>
#include <vector>

#include <raylib.h>

int main() {
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(0, 0, "Raytracer demo");
  Texture2D texture;
  uint32_t *framebuffer = 0;
  int prev_width = 0, prev_height = 0;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    int width = GetRenderWidth(), height = GetRenderHeight();
    if (prev_width != width || prev_height != height) {
      free(framebuffer);
      framebuffer = (uint32_t *)malloc(sizeof(uint32_t) * width * height);
      init(width, height);
      Image img = {
          .data = framebuffer,
          .width = width,
          .height = height,
          .mipmaps = 1,
          .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
      };
      if (prev_width)
        UnloadTexture(texture);
      texture = LoadTextureFromImage(img);
      prev_width = width, prev_height = height;
    }

    // std::vector<sphere> spheres = {{
    std::array<sphere, 2> spheres = {{
        sphere{{-1, 0, -3}, 2 * powf(sin(GetTime()), 2), 0xFFEE2222},
        sphere{{+1, 0, -3}, 1 * powf(sin(GetTime() * 2 + PI * 0.5f), 2), 0xFF11DDFF},
    }};

    doRender(framebuffer, width, height, spheres.data(), spheres.size());

    UpdateTexture(texture, framebuffer);
    DrawTexture(texture, 0, 0, WHITE);

    EndDrawing();
  }
  CloseWindow();
}
