#!/bin/sh
clang -fclangir raytracer.cu -o raytracer  -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -pthread -lm --cuda-gpu-arch=sm_86
