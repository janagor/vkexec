ARG VARIANT="noble"
FROM ubuntu:${VARIANT} AS base

ARG VARIANT
ARG GCC_VER="16"
ARG LLVM_VER="22"

RUN apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg software-properties-common wget apt-utils \
        git ninja-build python3 \
        pkg-config graphviz doxygen ccache cppcheck gcovr mold gdb

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        gcc-${GCC_VER} g++-${GCC_VER} && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VER} 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VER} 100 && \
    update-alternatives --install /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-${GCC_VER} 100 && \
    update-alternatives --install /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-${GCC_VER} 100

RUN wget -O /tmp/llvm-snapshot.gpg.key https://apt.llvm.org/llvm-snapshot.gpg.key && \
    gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg /tmp/llvm-snapshot.gpg.key && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/${VARIANT}/ llvm-toolchain-${VARIANT}-${LLVM_VER} main" \
      > /etc/apt/sources.list.d/llvm.list && \
    apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        clang-${LLVM_VER} clang-tools-${LLVM_VER} clang-tidy-${LLVM_VER} \
        lld-${LLVM_VER} llvm-${LLVM_VER} llvm-${LLVM_VER}-dev libclang-${LLVM_VER}-dev \
        llvm-${LLVM_VER}-tools && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VER} 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VER} 100 && \
    update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-${LLVM_VER} 100 && \
    update-alternatives --install /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-${LLVM_VER} 100 && \
    update-alternatives --install /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-${LLVM_VER} 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-${LLVM_VER} 100

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-latest.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-latest.gpg] https://apt.kitware.com/ubuntu/ ${VARIANT} main" \
      > /etc/apt/sources.list.d/kitware.list && \
    apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends cmake

ENV IWYU=/opt/iwyu
ENV IWYU_BUILD=${IWYU}/build
ENV IWYU_SRC=${IWYU}/include-what-you-use
RUN mkdir -p "${IWYU_BUILD}" && \
    git clone --depth 1 --branch clang_${LLVM_VER} \
        https://github.com/include-what-you-use/include-what-you-use.git \
        "${IWYU_SRC}" && \
    CC=clang-${LLVM_VER} CXX=clang++-${LLVM_VER} cmake -S "${IWYU_SRC}" -B "${IWYU_BUILD}" \
        -G Ninja -DCMAKE_PREFIX_PATH=/usr/lib/llvm-${LLVM_VER} && \
    cmake --build "${IWYU_BUILD}" -j && \
    cmake --install "${IWYU_BUILD}" && \
    mkdir -p "$(include-what-you-use -print-resource-dir 2>/dev/null)" && \
    ln -sf "$(readlink -f /usr/lib/clang/${LLVM_VER}/include)" \
        "$(include-what-you-use -print-resource-dir 2>/dev/null)/include"

RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

FROM base AS gcc
ENV CC=gcc
ENV CXX=g++
CMD ["/bin/bash"]

FROM base AS clang
ENV CC=clang
ENV CXX=clang++
CMD ["/bin/bash"]
