#include "header.h"

#include <cmath>
#include <cstdlib>

#include <array>
#include <vector>

#include <raylib.h>

int main() {
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(1800, 1800, "CIR CUDA Demo");
  Texture2D texture;
  Colour *framebuffer = 0;
  int prev_width = 0, prev_height = 0;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    int width = GetRenderWidth(), height = GetRenderHeight();
    if (prev_width != width || prev_height != height) {
      free(framebuffer);
      framebuffer = (Colour *)malloc(sizeof(uint32_t) * width * height);
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

    std::vector<sphere> spheres = {{
    // std::array<sphere, 1> spheres = {{
        // sphere{{0, 0, 3}, 1, Colour{255, 0, 0, 255}},
        sphere{{0, 0, 3}, 0.5f + 1.5f * powf(sin(GetTime() * 0.7f), 2), Colour{255, 0, 0, 255}},
        // sphere{{1, 1, 3}, 0.5, Colour{0, 0, 255, 255}},
        // sphere{{0, 2, 3}, 2, Colour{255, 10, 0, 255}},
        // sphere{{-1, 0, -3}, 2 * powf(sin(GetTime()), 2), Colour{0, 0, 255, 255}},
        // sphere{{+1, 0, -3}, 1.5f * powf(sin(GetTime() * 1.2 + PI * 0.5f), 2), Colour{255, 0, 0, 255}},
    }};

    doRender(framebuffer, width, height, spheres.data(), spheres.size());

    UpdateTexture(texture, framebuffer);
    DrawTexture(texture, 0, 0, WHITE);

    EndDrawing();
  }
  CloseWindow();
}
