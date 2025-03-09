#include "header.h"
#include <cmath>
#include <cstdlib>

#include <raylib.h>

int main() {
  constexpr int width = 512, height = 512;
  init(width, height);
  uint32_t *framebuffer = (uint32_t *)malloc(sizeof(uint32_t) * width * height);
  InitWindow(width, height, "Hello");
  Image img = {
      .data = framebuffer,
      .width = width,
      .height = height,
      .mipmaps = 1,
      .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
  };
  Texture2D texture = LoadTextureFromImage(img);
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    sphere spheres[] = {
        {{0, 0, -3}, 2 * powf(sin(GetTime()), 2), 0xFFFF0000},
        {{1, 0, -3}, 2 * powf(sin(GetTime() * 2 + 1), 2), 0xFF00FF00},
    };
    doRender(framebuffer, spheres, sizeof(spheres) / sizeof(sphere));

    DrawTexture(texture, 0, 0, WHITE);
    UpdateTexture(texture, framebuffer);

    EndDrawing();
  }
  UnloadTexture(texture);
  CloseWindow();
}
