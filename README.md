<p align="center">
  <img src="doc/logo/otter_icon.png" alt="OTTER logo" width="200">
</p>

<br clear="left"/>
<br clear="left"/>

[![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue?logo=docker)](https://ghcr.io/mschreter/otter)
[![Pre-commit](https://github.com/mschreter/otter/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/mschreter/otter/actions/workflows/pre-commit.yml)
[![GitHub CI](https://github.com/mschreter/Otter/actions/workflows/tests.yml/badge.svg)](https://github.com/mschreter/Otter/actions/workflows/tests.yml)

# From CT data to homogenized properties



This repository contains an efficient finite element implementation for solving the screened Poisson equation using the deal.II library.

## Running with Docker (recommended for users)

A prebuilt Docker image is available via GitHub Container Registry:

```bash
docker pull ghcr.io/mschreter/otter:latest
```

To run a simulation, mount your working directory into the container and execute OTTER:

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/mschreter/otter:latest \
  otter <my_input_file>
```

For MPI runs, use:

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/mschreter/otter:latest \
  mpirun -np <n_ranks> otter <my_input_file>
```

### Example

```bash
cd examples/spot-the-cow

docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/mschreter/otter:latest \
  mpirun -np 4 otter input_cp.json
```

All output files are written directly to the mounted working directory on the host system.

## Installation (for developers)

### 1. Install deal.II via Candi

First, install [deal.II](https://www.dealii.org/) using the [candi](https://github.com/dealii/candi) build system:

```bash
git clone https://github.com/dealii/candi.git
cd candi
```
Uncomment this line in `candi.cfg` to enable ArborX support ...
```bash
#PACKAGES="${PACKAGES} once:arborx"
```
... and install:
```bash
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
git clone https://github.com/mschreter/otter
cd otter
cmake --preset=release
cd build-release
make -j<n_processes>
```
Replace `<n_processes>` with the number of CPU threads you want to use for compilation.

## Running a simulation

We run e.g. the example `spot-the-cow` by executing

```bash
cd otter/examples/spot-the-cow
mpirun -np <n_processes> ../build-release/otter input_cp.json
```
