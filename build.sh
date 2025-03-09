#!/bin/sh
# Use your gpu-arch, for eg
# clang --cuda-gpu-arch=sm_75 -c raytracer.cu
clang --cuda-gpu-arch=sm_86 -c raytracer.cu
clang -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -lm -lpthread -lraylib raytracer.o main.cpp -o demo
