#!/bin/python3
import os
import subprocess
import shutil


def shell(command):
    return subprocess.check_output(command, shell=True).decode("utf-8").strip()


# CAPITAL variables are configurations.

# Change this to your corresponding ld-....so.
LIBCSO = "/lib64/ld-linux-x86-64.so.2"

# Compute capability; eg. for sm_75, it would return "7.5".
cap = shell("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
SM = f"sm_{cap.replace('.', '')}"

LL_BUILD_DIR = shell("llvm-config --prefix")
LLVM_DIR = os.path.dirname(LL_BUILD_DIR)
CUDA_DIR = "/usr/local/cuda/include"

# Building directory for this repository.
BUILD_DIR = "build"

# Note: the order is important.
# Changing the order causes mysterious failures.
SYSTEM_DIRS = [
    "/usr/include/c++/13",
    "/usr/include/c++/14",
    "/usr/include/x86_64-linux-gnu/c++/13",
    "/usr/include/c++/14/x86_64-redhat-linux",
    "/usr/include/c++/13/backward",
    f"{LL_BUILD_DIR}/lib/clang/20/include",
    "/usr/local/include",
    "/usr/include/x86_64-linux-gnu",
    "/usr/include",
    CUDA_DIR,
]

EXTERNC_DIRS = [
    # Standard c headers
    "/usr/include/x86_64-linux-gnu",
    "/usr/include",
    "/include",
    "/usr/local/include",
]

LIBPATHS = [
    "/usr/local/cuda/lib64",
    "/usr/lib/gcc/x86_64-linux-gnu/13",
    "/lib/gcc/x86_64-redhat-linux/14",
    "/usr/lib64",
    "/lib/x86_64-linux-gnu",
    "/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/lib",
    "/usr/lib",
    "raylib/lib",
]

LIBRARIES = [
    "cudart_static",
    "dl",
    "rt",
    "pthread",
    "raylib",
    "m",
    "stdc++",
    "gcc_s",
    "gcc",
    "c",
]

DEVICE_TRIPLE = "nvptx64-nvidia-cuda"
HOST_TRIPLE = "x86_64-unknown-linux-gnu"


def sysd(dir):
    return f"-internal-isystem {dir}"


def c(dir):
    return f"-internal-externc-isystem {dir}"


CXXFLAGS = [
    "-fclangir",
    "-mlink-builtin-bitcode /usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
    f"-resource-dir {LL_BUILD_DIR}/lib/clang/20",
    f"-internal-isystem {LL_BUILD_DIR}/lib/clang/20/include/cuda_wrappers",
    "-include __clang_cuda_runtime_wrapper.h",
    " ".join([f"-internal-isystem {dir}" for dir in SYSTEM_DIRS]),
    " ".join([f"-internal-externc-isystem {dir}" for dir in EXTERNC_DIRS]),
    "-fgnuc-version=4.2.1",
    "-D__GCC_HAVE_DWARF2_CFI_ASM=1",
    "-x cuda",
]

# CUID isn't important.
# It's ok as long as it's unique, and it's the same for host and device.
CUID = "7f271ca5beb48ee9"

DEVICEFLAGS = [
    "-S",
    "-target-sdk-version=12.8",
    "-fcuda-is-device",
    f"-cuid={CUID}",
    f"-target-cpu {SM}",
    f"-aux-target-cpu x86-64",
    f"-triple {DEVICE_TRIPLE}",
    f"-aux-triple {HOST_TRIPLE}",
]

HOSTFLAGS = [
    "-fclangir-call-conv-lowering",
    "-emit-obj",
    "-pic-level 2",
    "-pic-is-pie",
    "-target-sdk-version=12.8",
    f"-cuid={CUID}",
    f"-fcuda-include-gpubinary {BUILD_DIR}/device.fatbin",
    f"-target-cpu x86-64",
    f"-aux-target-cpu {SM}",
    f"-triple {HOST_TRIPLE}",
    f"-aux-triple {DEVICE_TRIPLE}",
]

if os.environ.get("FEDORA"):
    CRUNTIME = [
        "/lib64/Scrt1.o",
        "/lib64/crti.o",
        "/lib/gcc/x86_64-redhat-linux/14/crtbeginS.o",
        "/lib/gcc/x86_64-redhat-linux/14/crtendS.o",
        "/lib64/crtn.o",
    ]
else:
    CRUNTIME = [
        "/lib/x86_64-linux-gnu/Scrt1.o",
        "/lib/x86_64-linux-gnu/crti.o",
        "/usr/lib/gcc/x86_64-linux-gnu/13/crtbeginS.o",
        "/usr/lib/gcc/x86_64-linux-gnu/13/crtendS.o",
        "/lib/x86_64-linux-gnu/crtn.o",
    ]

LDFLAGS = [
    "-z relro",
    "--hash-style=gnu",
    "--eh-frame-hdr",
    "-m elf_x86_64",
    "-pie",
    f"-dynamic-linker {LIBCSO}",
    " ".join(CRUNTIME),
]

cxxflags = " ".join(CXXFLAGS)

# Create build folder if that's not available.
os.makedirs(BUILD_DIR, exist_ok=True)
shutil.rmtree(BUILD_DIR)
os.makedirs(BUILD_DIR, exist_ok=True)

# Compile device-side code.
deviceflags = " ".join(DEVICEFLAGS)
print(

    f"clang -cc1 {deviceflags} {cxxflags} -o {BUILD_DIR}/device.s raytracer.cu",
)
subprocess.run(
    f"clang -cc1 {deviceflags} {cxxflags} -o {BUILD_DIR}/device.s raytracer.cu",
    shell=True,
    check=True,
    stderr = subprocess.DEVNULL,
)

# Assemble PTX into cubin.
print(

    f"ptxas -m64 -O0 --gpu-name {SM} --output-file {BUILD_DIR}/device.cubin {BUILD_DIR}/device.s",
)
subprocess.run(
    f"ptxas -m64 -O0 --gpu-name {SM} --output-file {BUILD_DIR}/device.cubin {BUILD_DIR}/device.s",
    shell=True,
    check=True,
    stderr = subprocess.DEVNULL,
)

# Embed cubin into a fat binary.
print(

    f"fatbinary -64 --create {BUILD_DIR}/device.fatbin --image=profile={SM},file={BUILD_DIR}/device.cubin",
)
subprocess.run(
    f"fatbinary -64 --create {BUILD_DIR}/device.fatbin --image=profile={SM},file={BUILD_DIR}/device.cubin",
    shell=True,
    check=True,
    stderr = subprocess.DEVNULL,
)

# Compile host-side code.
hostflags = " ".join(HOSTFLAGS)
print(

    f"clang++ -cc1 {hostflags} {cxxflags} -o {BUILD_DIR}/host.o raytracer.cu",
)
subprocess.run(
    f"clang++ -cc1 {hostflags} {cxxflags} -o {BUILD_DIR}/host.o raytracer.cu",
    shell=True,
    check=True,
    stderr = subprocess.DEVNULL,
    stdout = subprocess.DEVNULL,
)

# Compile `main.cpp` normally.
print(

    f"clang++ -c -Iraylib/include -o {BUILD_DIR}/main.o main.cpp",
)
subprocess.run(
    f"clang++ -c -Iraylib/include -o {BUILD_DIR}/main.o main.cpp",
    shell=True,
    check=True,
    stderr = subprocess.DEVNULL,
    stdout = subprocess.DEVNULL,
)

# Link with CUDA.
libpaths = " ".join([f"-L{path}" for path in LIBPATHS])
libs = " ".join([f"-l{lib}" for lib in LIBRARIES])
ldflags = " ".join(LDFLAGS)
print(

    f"ld {ldflags} -o {BUILD_DIR}/raytracer {libpaths} {BUILD_DIR}/host.o {BUILD_DIR}/main.o {libs}",
)
subprocess.run(
    f"ld {ldflags} -o {BUILD_DIR}/raytracer {libpaths} {BUILD_DIR}/host.o {BUILD_DIR}/main.o {libs}",
    shell=True,
    check=True,
    stderr = subprocess.DEVNULL,
    stdout = subprocess.DEVNULL,
)
