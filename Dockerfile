FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# Default is slow archive source. Swap it with normal ubuntu source.
RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|mirror://mirrors.ubuntu.com/mirrors.txt|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|mirror://mirrors.ubuntu.com/mirrors.txt|g' \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libboost-dev \
        libblas-dev \
        libgmp-dev \
        liblapack-dev \
        libmpfr-dev \
        libopenmpi-dev \
        libssl-dev \
        mold \
        ninja-build \
        openmpi-bin \
        pkg-config \
        xz-utils \
        libsuitesparse-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/polysolve

ARG POLYSOLVE_COMMIT
RUN git clone --branch hybrid-dev --single-branch https://github.com/iiiian/polysolve.git src \
    && cd src \
    && git checkout "${POLYSOLVE_COMMIT}"

WORKDIR /opt/polysolve/src

ENV CPM_SOURCE_CACHE=/opt/cpm_cache

RUN cmake -S . -B ./build/release -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold -Wl,--no-as-needed -lmpi_cxx" \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=mold \
        -DPOLYSOLVE_WITH_CUDA=ON \
        -DPOLYSOLVE_WITH_HYPRE=ON \
        -DPOLYSOLVE_WITH_TESTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES=all-major \
        -DPOLYSOLVE_WITH_ICHOL=ON

RUN cmake --build ./build/release

# Ensure non-root Singularity users can read downloaded CPM libraries
RUN chmod -R a+rX /opt/cpm_cache

ENTRYPOINT ["/opt/polysolve/src/build/release/tests/linear_solve"]
