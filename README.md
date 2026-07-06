# 6004CEM Parallel & Distributed Computing Coursework

This repository contains the code for the 6004CEM coursework. It is split into two parts:

- **Part 1** – OpenMP parallel matrix multiplication benchmark (speedup, efficiency, Amdahl/Gustafson analysis).
- **Part 2** – K-Means clustering: a sequential baseline and a distributed MPI implementation using a star/master-worker topology.

All build and run commands are wrapped in a [`justfile`](justfile).

---

## Requirements

These programs were developed and tested on **macOS** with the following tools installed:

- `clang++` (with C++17 support)
- `just` – command runner (`brew install just`)
- `libomp` – OpenMP runtime for Part 1 (`brew install libomp`)
- Open MPI – for Part 2 MPI runs (`brew install open-mpi`)

If you are on Linux, the `clang++` OpenMP flags can be simplified to `-fopenmp` (paths in the justfile may need adjustment).

---

## Quick Start

Clone the repository and run any of the `just` recipes from the project root:

```bash
cd 6004CEM

# Part 1: OpenMP matrix multiplication benchmark
just one

# Part 2: sequential K-Means only
just two

# Part 2: MPI K-Means only (2, 4, 6, 8 processes)
just mpi

# Part 2: full benchmark (sequential + MPI) and produce performance_part2.csv
just part2
```

---

## Part 1 – OpenMP Matrix Multiplication

Source: [`Part 1/solution.cpp`](Part%201/solution.cpp)

This program benchmarks dense `n × n` matrix multiplication using three variants:

1. Sequential baseline
2. OpenMP `parallel for` with static scheduling
3. OpenMP task-based parallelism with 64×64 blocks

It tests matrix sizes `200, 500, 1000, 1500` and thread counts `1, 2, 4, 8` (plus the hardware maximum), and writes results to `Part 1/performance_results.csv`.

### Run Part 1

```bash
just one
```

Or manually:

```bash
cd "Part 1"
clang++ -std=c++17 -O2 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib \
    -lomp solution.cpp -o solution
./solution
```

### Output

- Console tables with mean time, speedup, efficiency, Amdahl parallel fraction, and Gustafson serial fraction for each variant.
- `Part 1/performance_results.csv` for plotting speedup/efficiency graphs.

---

## Part 2 – K-Means Clustering

Source: [`Part 2/solution.cpp`](Part%202/solution.cpp)

The same source file compiles into two executables:

- `solution` – sequential baseline (no preprocessor flags).
- `solution_mpi` – distributed K-Means using MPI (`-DUSE_MPI`).

The default dataset has `N = 500000` points, `D = 2` dimensions, and `K = 5` clusters. Edit the `Config` struct in the source to change these values.

### Run the sequential baseline

```bash
just two
```

Or manually:

```bash
cd "Part 2"
clang++ -std=c++17 -O2 solution.cpp -o solution
./solution
```

The sequential run writes its final centroids to `centroids_seq.txt`, which the MPI run uses for correctness comparison.

### Run the MPI implementation

```bash
just mpi
```

Or manually:

```bash
cd "Part 2"
export OMPI_MCA_btl=self,sm   # shared-memory transport only (avoids macOS firewall prompts)
mpicxx -std=c++17 -O2 -DUSE_MPI solution.cpp -o solution_mpi
mpirun --oversubscribe -np 4 ./solution_mpi
```

The `just mpi` recipe runs the MPI executable with `2, 4, 6, 8` processes and prints the last line of each run.

### Run the full Part 2 benchmark

```bash
just part2
```

This command:

1. Runs the sequential baseline.
2. Builds the MPI executable.
3. Runs MPI with `2, 4, 6, 8` processes.
4. Produces `Part 2/performance_part2.csv` with columns:
   `Processes, Version, MeanTime_ms, StdDevTime_ms, DataGen_ms, Iterations, CommTime_ms, CompTime_ms, CommRatio, BytesSent, BytesRecv, MaxCentroidError, Speedup`

---

## Project Structure

```
.
├── justfile                 # Build/run recipes
├── README.md                # This file
├── Part 1/
│   ├── solution.cpp         # OpenMP matrix multiplication
│   └── solution             # Pre-built binary
└── Part 2/
    ├── solution.cpp         # K-Means (sequential + MPI)
    ├── solution             # Pre-built sequential binary
    └── solution_mpi         # Pre-built MPI binary
```

---

## Notes

- The justfile sets `OMPI_MCA_btl=self,sm` before `mpirun` to force shared-memory transport and suppress repeated macOS firewall permission dialogs.
- Pre-built binaries are included for convenience but can be rebuilt from source using the commands above.
- Results are deterministic because both parts use fixed random seeds.
