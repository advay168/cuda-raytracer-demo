all: raytracer

build/raytracer : main.cpp raytracer.cu header.h compile.py
	./compile.py

raytracer: build/raytracer
	build/raytracer
