#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

class Matrix {
  size_t n;
  std::vector<double> data;

public:
  explicit Matrix(size_t size) : n(size), data(size * size, 0.0) {}

  inline double &at(size_t i, size_t j) { return data[i * n + j]; }
  inline const double &at(size_t i, size_t j) const { return data[i * n + j]; }
  inline size_t size() const { return n; }

  void randomize() {
    std::mt19937 gen(42); // fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 10.0);
    for (auto &val : data)
      val = dist(gen);
  }

  void zero() { std::fill(data.begin(), data.end(), 0.0); }
};

// -----------------------------------------------------------------------------
// 1. Sequential Baseline
// -----------------------------------------------------------------------------
void multiply_sequential(const Matrix &A, const Matrix &B, Matrix &C, int n) {
  C.zero();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += A.at(i, k) * B.at(k, j);
      }
      C.at(i, j) = sum;
    }
  }
}

// -----------------------------------------------------------------------------
// 2. OpenMP Parallel-For
// -----------------------------------------------------------------------------
void multiply_parallel(const Matrix &A, const Matrix &B, Matrix &C, int n) {
  C.zero();
  // schedule(static) -> divides work into fixed-size chunks at compile time
#pragma omp parallel for schedule(static) default(none) shared(A, B, C, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += A.at(i, k) * B.at(k, j);
      }
      C.at(i, j) = sum;
    }
  }
}

// -----------------------------------------------------------------------------
// 3. Task-Based Parallelism
// The runtime work-stealing scheduler balances load across threads dynamically.
// -----------------------------------------------------------------------------
void multiply_tasks(const Matrix &A, const Matrix &B, Matrix &C, int n,
                    int block_size = 64) {
  C.zero();
#pragma omp parallel default(none) shared(A, B, C, n, block_size)
  {
#pragma omp single
    {
      for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
          // Each task computes one block of C independently.
          // firstprivate captures ii/jj by value -> no race condition (threads
          // take on tasks at any time).
#pragma omp task firstprivate(ii, jj) default(none)                            \
    shared(A, B, C, n, block_size)
          {
            int i_max = std::min(ii + block_size, n);
            int j_max = std::min(jj + block_size, n);
            for (int i = ii; i < i_max; ++i) {
              for (int j = jj; j < j_max; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                  sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) = sum;
              }
            }
          }
        }
      }
      // Implicit barrier at end of single block waits for all tasks.
    }
  }
}

// Wrapper so task version matches the same signature for benchmarking
void multiply_tasks_wrapper(const Matrix &A, const Matrix &B, Matrix &C,
                            int n) {
  multiply_tasks(A, B, C, n, 64);
}

// -----------------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------------
bool verify_equal(const Matrix &C1, const Matrix &C2, int n,
                  double tol = 1e-9) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (std::abs(C1.at(i, j) - C2.at(i, j)) > tol) {
        std::cerr << "  Mismatch at (" << i << "," << j << ")\n";
        return false;
      }
    }
  }
  return true;
}

// -----------------------------------------------------------------------------
// Statistics & Theory Helpers
// -----------------------------------------------------------------------------
struct TimingStats {
  double mean;
  double stddev;
  double min_time;
  double max_time;
};

TimingStats compute_stats(const std::vector<double> &times) {
  TimingStats s;
  s.min_time = *std::min_element(times.begin(), times.end());
  s.max_time = *std::max_element(times.begin(), times.end());
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  s.mean = sum / times.size();
  double sq = 0.0;
  for (double t : times)
    sq += (t - s.mean) * (t - s.mean);
  s.stddev = std::sqrt(sq / times.size());
  return s;
}

// Amdahl's Law:  S = 1 / ((1-P) + P/N)  ->  P = (1 - 1/S) / (1 - 1/N)
double amdahl_parallel_fraction(double speedup, int n_threads) {
  if (speedup <= 0.0 || n_threads <= 1)
    return 0.0;
  double denom = 1.0 - (1.0 / n_threads);
  if (denom < 1e-12)
    return 0.0;
  return (1.0 - (1.0 / speedup)) / denom;
}

// Gustafson's Law:  S = N - α(N-1)  ->  α = (N - S) / (N - 1)
double gustafson_serial_fraction(double speedup, int n_threads) {
  if (n_threads <= 1)
    return 1.0;
  return (n_threads - speedup) / (n_threads - 1.0);
}

// -----------------------------------------------------------------------------
// System Info
// -----------------------------------------------------------------------------
void print_system_info() {
  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) == 0) {
    std::cout << "VM Hostname: " << hostname << "\n";
  }
  struct utsname u;
  if (uname(&u) == 0) {
    std::cout << "Operating System:  " << u.sysname << " " << u.release << "\n";
    std::cout << "Architecture:    " << u.machine << "\n";
  }
  std::cout << "CPU Cores:       " << omp_get_num_procs() << "\n";
  std::cout << "Max OMP Threads: " << omp_get_max_threads() << "\n";
  std::cout << "Compiler:        GCC " << __GNUC__ << "." << __GNUC_MINOR__
            << "\n";
  std::cout << "C++ Standard:    " << __cplusplus << "\n";
  std::cout
      << "================================================================\n\n";
}

// -----------------------------------------------------------------------------
// Benchmarking Engine
// -----------------------------------------------------------------------------
using MultiplyFunc =
    std::function<void(const Matrix &, const Matrix &, Matrix &, int)>;

struct BenchmarkResult {
  std::string version;
  int threads;
  TimingStats stats;
  double speedup;
  double efficiency;
  double amdahl_p;
  double gustafson_alpha;
};

std::vector<BenchmarkResult>
benchmark_version(const std::string &name, MultiplyFunc func, const Matrix &A,
                  const Matrix &B, Matrix &C, int n,
                  const std::vector<int> &thread_counts, double seq_mean,
                  int warmup, int measured) {
  std::vector<BenchmarkResult> out;
  for (int nt : thread_counts) {
    if (nt < 1)
      continue;
    omp_set_num_threads(nt);

    // Warm-up: cache + thread pool initialization
    for (int w = 0; w < warmup; ++w) {
      C.zero();
      func(A, B, C, n);
    }

    // Measured runs
    std::vector<double> times;
    times.reserve(measured);
    for (int m = 0; m < measured; ++m) {
      C.zero();
      double t0 = omp_get_wtime();
      func(A, B, C, n);
      double t1 = omp_get_wtime();
      times.push_back(t1 - t0);
    }

    TimingStats st = compute_stats(times);
    double speedup = seq_mean / st.mean;
    double efficiency = speedup / nt;
    out.push_back({name, nt, st, speedup, efficiency,
                   amdahl_parallel_fraction(speedup, nt),
                   gustafson_serial_fraction(speedup, nt)});
  }
  return out;
}

void print_table_header() {
  std::cout << std::left << std::setw(16) << "Version" << std::setw(6) << "Thr"
            << std::setw(12) << "Time(s)" << std::setw(12) << "StdDev"
            << std::setw(10) << "Speedup" << std::setw(12) << "Efficiency"
            << std::setw(12) << "Amdahl-P" << std::setw(12) << "Gustafson-a"
            << "\n";
  std::cout << std::string(90, '-') << "\n";
}

void print_row(const BenchmarkResult &r) {
  std::cout << std::left << std::setprecision(4) << std::fixed << std::setw(16)
            << r.version << std::setw(6) << r.threads << std::setw(12)
            << r.stats.mean << std::setw(12) << r.stats.stddev << std::setw(10)
            << std::setprecision(2) << r.speedup << std::setw(12)
            << std::setprecision(2) << r.efficiency << std::setw(12)
            << std::setprecision(4) << r.amdahl_p << std::setw(12)
            << std::setprecision(4) << r.gustafson_alpha << "\n";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main() {
  std::cout
      << "================================================================\n";
  std::cout << "  6004CEM Performance Benchmark\n";
  std::cout
      << "================================================================\n";
  print_system_info();

  // CSV for report graphs
  std::ofstream csv("performance_results.csv");
  csv << "Size,Version,Threads,MeanTime,StdDev,MinTime,Speedup,Efficiency,"
      << "Amdahl_P,Gustafson_Alpha\n";

  // --- Experimental Design ---
  // Thread counts: powers of 2 up to hardware limit
  std::vector<int> thread_counts = {1, 2, 4, 8};
  int max_t = omp_get_max_threads();
  if (std::find(thread_counts.begin(), thread_counts.end(), max_t) ==
      thread_counts.end())
    thread_counts.push_back(max_t);
  std::sort(thread_counts.begin(), thread_counts.end());
  thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()),
                      thread_counts.end());

  std::vector<int> sizes = {200, 500, 1000, 1500};
  const int WARMUP = 2;
  const int MEASURED = 7;

  for (int n : sizes) {
    long long ops = static_cast<long long>(n) * n * n;
    std::cout << "\nMATRIX SIZE: " << n << "x" << n << " (" << ops / 1'000'000
              << "M floating-point ops)\n";
    std::cout
        << "================================================================\n";

    Matrix A(n), B(n), C_seq(n), C_par(n), C_task(n);
    A.randomize();
    B.randomize();

    // --- Sequential Baseline (stable mean, not just one run) ---
    std::vector<double> seq_times;
    for (int m = 0; m < WARMUP + MEASURED; ++m) {
      C_seq.zero();
      double t0 = omp_get_wtime();
      multiply_sequential(A, B, C_seq, n);
      double t1 = omp_get_wtime();
      if (m >= WARMUP)
        seq_times.push_back(t1 - t0);
    }
    TimingStats seq_stats = compute_stats(seq_times);
    std::cout << "Sequential baseline: " << seq_stats.mean << " s (+/- "
              << seq_stats.stddev << " s)\n\n";

    print_table_header();

    // Print sequential reference row
    BenchmarkResult seq_ref{"Sequential", 1, seq_stats, 1.0, 1.0, 0.0, 1.0};
    print_row(seq_ref);
    csv << n << ",Sequential,1," << seq_stats.mean << "," << seq_stats.stddev
        << "," << seq_stats.min_time << ",1.0,1.0,0.0,1.0\n";

    // --- Parallel-For ---
    auto r_par =
        benchmark_version("Parallel-For", multiply_parallel, A, B, C_par, n,
                          thread_counts, seq_stats.mean, WARMUP, MEASURED);
    if (!verify_equal(C_seq, C_par, n))
      std::cout << "  ERROR: Parallel-For result verification FAILED!\n";

    for (const auto &r : r_par) {
      print_row(r);
      csv << n << ",Parallel-For," << r.threads << "," << r.stats.mean << ","
          << r.stats.stddev << "," << r.stats.min_time << "," << r.speedup
          << "," << r.efficiency << "," << r.amdahl_p << ","
          << r.gustafson_alpha << "\n";
    }

    // --- Task-Based ---
    auto r_task =
        benchmark_version("Task-Based", multiply_tasks_wrapper, A, B, C_task, n,
                          thread_counts, seq_stats.mean, WARMUP, MEASURED);
    if (!verify_equal(C_seq, C_task, n))
      std::cout << "  ERROR: Task-Based result verification FAILED!\n";

    for (const auto &r : r_task) {
      print_row(r);
      csv << n << ",Task-Based," << r.threads << "," << r.stats.mean << ","
          << r.stats.stddev << "," << r.stats.min_time << "," << r.speedup
          << "," << r.efficiency << "," << r.amdahl_p << ","
          << r.gustafson_alpha << "\n";
    }

    // --- Bottleneck & Behaviour Notes ---
    std::cout << "\n-- Observations --\n";
    if (n <= 300)
      std::cout << "* Small n: Thread management overhead dominates; parallel "
                   "may be slower.\n";
    if (n >= 1000)
      std::cout << "* Large n: Memory bandwidth contention limits super-linear "
                   "speedup.\n";
    if (!r_par.empty() && r_par.back().efficiency < 0.5)
      std::cout << "* Efficiency < 50% at max threads: barrier/sync overhead "
                   "is significant.\n";
    std::cout << "* Static scheduling used for uniform loop iterations (good "
                 "load balance).\n";
    std::cout << "* Task version incurs creation overhead but enables dynamic "
                 "work-stealing.\n";
    std::cout << "* Race condition: none (each C(i,j) written exactly once, no "
                 "reduction).\n";
    std::cout << "* False sharing: low risk (blocks are 64x64 = 512 doubles = "
                 "4KB, > cache line).\n";
  }

  csv.close();
  std::cout << "\n\nCSV written to: performance_results.csv\n";
  std::cout << "Import into Excel/Python/MATLAB to generate Speedup & "
               "Efficiency graphs.\n";
  return 0;
}
