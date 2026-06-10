# Screened Poisson Solver (Efficient FEM)

This repository contains an efficient finite element implementation for solving the screened Poisson equation using the deal.II library.

## Installation (without ArborX)

### 1. Install deal.II via Candi

First, install [deal.II](https://www.dealii.org/) using the [candi](https://github.com/dealii/candi) build system:

```bash
git clone https://github.com/dealii/candi.git
cd candi
./candi.sh
```
Once installed, export the deal.II environment for your current shell session:
```
source /path/to/dealii-candi/configuration/deal.II-master
```
Replace `/path/to/dealii-candi` with the actual path on your machine.

### 2. Install this project
Clone the repository and build the solver:
```bash
git clone https://github.com/mschreter/screened-poisson-efficient
cd screened-poisson-efficient/fem
cmake --preset=release
cd build-release
make -j<n_processes>
```
Replace `<n_processes>` with the number of CPU threads you want to use for compilation.

## Running a simulation

We run e.g. the example placed in the tests folder:

```bash
cd screened-poisson-efficient/tests
mpirun -np <n_processes> ../build-release/screened-poisson-efficient input_abaqus.json
```

## Installation with ArborX

Make sure the replaces `n_processes` with a corresponding value

```bash
# install spack
git clone https://github.com/spack/spack ~/.spack/Spack
. ~/.spack/Spack/share/spack/setup-env.sh
echo '. ~/.spack/Spack/share/spack/setup-env.sh' >> ~/.bash_profile

# download the repository
git clone https://github.com/mschreter/screened-poisson-efficient
cd screened-poisson-efficient/fem

# install dependencies via spack
spack env create -d spe spack.yaml
spack env activate spe
spack concretize
spack install

# install deal.II into screened-poisson-efficient/fem/build-dealii
git clone https://github.com/dealii/dealii.git
mkdir -p build-dealii
cd build-dealii
cmake \
    -D CMAKE_CXX_COMPILER="mpicxx" \
    -D CMAKE_CXX_FLAGS="-march=native -Wno-array-bounds" \
    -D CMAKE_BUILD_TYPE="Release" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D CMAKE_CXX_STANDARD="20" \
    -D CMAKE_C_COMPILER="mpicc" \
    -D MPIEXEC_PREFLAGS="-bind-to none" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D DEAL_II_WITH_PETSC="OFF" \
    -D DEAL_II_WITH_METIS="ON" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_TRILINOS="ON" \
    -D DEAL_II_WITH_ZLIB="ON" \
    -D DEAL_II_FORCE_BUNDLED_BOOST="ON" \
    -D DEAL_II_WITH_KOKKOS="ON" \
    -D DEAL_II_WITH_ARBORX="ON" \
    -D DEAL_II_WITH_HDF5="OFF" \
    -D DEAL_II_WITH_64BIT_INDICES="OFF" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    ../dealii
make -j<n_processes>
cd ..
mkdir -p build
cd build
cmake -DDEAL_II_DIR=build-dealii -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Release ..
make -j<n_processes>
```
