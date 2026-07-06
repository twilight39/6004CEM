#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#ifdef USE_MPI
#include <cstring>
#include <mpi.h>
#endif

using namespace std;

// K-Means Clustering configuration.
struct Config {
  int N = 500000; // Total points
  int D = 2;      // Dimensions (keep small: 2 or 3)
  int K = 5;      // Clusters
  int max_iters = 100;
  double epsilon = 1e-4; // Convergence threshold
  unsigned seed = 42;    // Fixed seed for reproducible benchmarks
};

using Point = vector<double>;
using Dataset = vector<Point>;

const int REPEATS = 5;

// ---------------------------------------------------------------------------
// High-resolution timer
// ---------------------------------------------------------------------------
class Timer {
  std::chrono::high_resolution_clock::time_point start;

public:
  Timer() : start(std::chrono::high_resolution_clock::now()) {}
  double elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }
};

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------
struct Stats {
  double mean = 0.0;
  double stddev = 0.0;
};

Stats compute_stats(const vector<double> &times) {
  Stats s;
  size_t n = times.size();
  if (n == 0)
    return s;

  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  s.mean = sum / static_cast<double>(n);
  double sq = 0.0;
  for (double t : times)
    sq += (t - s.mean) * (t - s.mean);
  s.stddev = std::sqrt(sq / static_cast<double>(n));
  return s;
}

// ---------------------------------------------------------------------------
// Squared Euclidean distance
// ---------------------------------------------------------------------------
double squared_distance(const Point &A, const Point &B) {
  double sum = 0.0;
  for (size_t i = 0; i < A.size(); ++i) {
    double diff = A[i] - B[i];
    sum += diff * diff;
  }
  return sum;
}

// ---------------------------------------------------------------------------
// Synthetic dataset generation
// ---------------------------------------------------------------------------
Dataset generate_dataset(const Config &cfg) {
  std::mt19937 rng(cfg.seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::uniform_int_distribution<int> cluster_pick(0, cfg.K - 1);

  std::vector<Point> cluster_centers(cfg.K, Point(cfg.D));
  for (int k = 0; k < cfg.K; ++k) {
    for (int d = 0; d < cfg.D; ++d) {
      cluster_centers[k][d] = (k * 10.0) + gauss(rng) * 2.0;
    }
  }

  Dataset data(cfg.N, Point(cfg.D));
  for (int i = 0; i < cfg.N; ++i) {
    int k = cluster_pick(rng);
    for (int d = 0; d < cfg.D; ++d) {
      data[i][d] = cluster_centers[k][d] + gauss(rng) * 1.5;
    }
  }

  std::shuffle(data.begin(), data.end(), rng);
  return data;
}

// ---------------------------------------------------------------------------
// Deterministic empty-cluster re-initialisation (replaces std::rand)
// ---------------------------------------------------------------------------
Point deterministic_reinit(const Dataset &data, int N, unsigned seed, int k,
                           int iter) {
  std::mt19937 rng(seed + static_cast<unsigned>(iter) * 1009u +
                   static_cast<unsigned>(k));
  std::uniform_int_distribution<int> dist(0, N - 1);
  return data[dist(rng)];
}

// ---------------------------------------------------------------------------
// Sequential baseline
// ---------------------------------------------------------------------------
tuple<vector<int>, vector<Point>, int> sequential_kmeans(const Dataset &data,
                                                         const Config &cfg) {
  int N = cfg.N;
  int D = cfg.D;
  int K = cfg.K;

  std::vector<Point> centroids(data.begin(), data.begin() + K);
  std::vector<int> assignments(N, -1);
  std::vector<Point> new_centroids(K, Point(D, 0.0));
  std::vector<int> counts(K, 0);

  int iter = 0;
  bool converged = false;

  for (; iter < cfg.max_iters && !converged; ++iter) {
    for (int k = 0; k < K; ++k) {
      // Reset new centroid matrix
      std::fill(new_centroids[k].begin(), new_centroids[k].end(), 0.0);
      counts[k] = 0;
    }

    // For each point
    for (int i = 0; i < N; ++i) {
      double best_dist = std::numeric_limits<double>::max();
      int best_k = -1;
      // Find the closest cluster centroid
      for (int k = 0; k < K; ++k) {
        double d = squared_distance(data[i], centroids[k]);
        if (d < best_dist) {
          best_dist = d;
          best_k = k;
        }
      }
      // Reassign it and increment the count
      assignments[i] = best_k;
      counts[best_k]++;
      // Add the point coordinates so the centroid can be recentered later
      for (int d = 0; d < D; ++d) {
        new_centroids[best_k][d] += data[i][d];
      }
    }

    double max_shift = 0.0;
    for (int k = 0; k < K; ++k) {
      // If cluster has 0 points, deterministically randomize it
      if (counts[k] == 0) {
        new_centroids[k] = deterministic_reinit(data, N, cfg.seed, k, iter);
        counts[k] = 1;
      }

      // Else, recenter the centroid
      for (int d = 0; d < D; ++d) {
        new_centroids[k][d] /= counts[k];
        double diff = std::abs(new_centroids[k][d] - centroids[k][d]);
        if (diff > max_shift)
          max_shift = diff;
      }
    }

    converged = (max_shift < cfg.epsilon);
    centroids.swap(new_centroids);
  }

  return {assignments, centroids, iter};
}

// ---------------------------------------------------------------------------
// Centroid file I/O (used to compare MPI vs sequential results)
// ---------------------------------------------------------------------------
void write_centroids(const vector<Point> &centroids, const string &path) {
  std::ofstream out(path);
  out << std::fixed << std::setprecision(6);
  for (const auto &c : centroids) {
    for (size_t i = 0; i < c.size(); ++i) {
      if (i)
        out << " ";
      out << c[i];
    }
    out << "\n";
  }
}

vector<Point> read_centroids(const string &path, int K, int D) {
  std::ifstream in(path);
  vector<Point> centroids;
  if (!in)
    return centroids;
  for (int k = 0; k < K; ++k) {
    Point p(D);
    for (int d = 0; d < D; ++d) {
      if (!(in >> p[d]))
        return {};
    }
    centroids.push_back(p);
  }
  return centroids;
}

#ifdef USE_MPI
// ---------------------------------------------------------------------------
// MPI helpers: flatten a Dataset into a contiguous buffer and back
// ---------------------------------------------------------------------------
static vector<double> flatten(const Dataset &data, int D) {
  vector<double> flat;
  flat.reserve(data.size() * D);
  for (const auto &p : data)
    flat.insert(flat.end(), p.begin(), p.end());
  return flat;
}

static Dataset unflatten(const vector<double> &flat, int local_N, int D) {
  Dataset data(local_N, Point(D));
  for (int i = 0; i < local_N; ++i) {
    for (int d = 0; d < D; ++d) {
      data[i][d] = flat[i * D + d];
    }
  }
  return data;
}

// ---------------------------------------------------------------------------
// Per-run instrumentation collected by the MPI implementation
// ---------------------------------------------------------------------------
struct MPIRunStats {
  double total_ms = 0.0;
  double comm_ms = 0.0;
  double comp_ms = 0.0;
  double scatter_ms = 0.0;
  double iter_comm_ms = 0.0;
  double gather_ms = 0.0;
  long long bytes_sent = 0;
  long long bytes_recv = 0;
  int iters = 0;
};

// ---------------------------------------------------------------------------
// Distributed K-Means using a STAR (master-worker) topology
// ---------------------------------------------------------------------------
class MPIStarKMeans {
  const Config &cfg_;
  MPI_Comm comm_;
  int rank_;
  int size_;

  // Partition information set by scatter_data()
  vector<int> counts_;
  vector<int> displs_;
  int local_N_ = 0;

  // Per-run byte counters
  long long bytes_sent_local_ = 0;
  long long bytes_recv_local_ = 0;

public:
  MPIStarKMeans(const Config &cfg, MPI_Comm comm) : cfg_(cfg), comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  }

  // Star scatter: master sends a contiguous chunk to every worker.
  void scatter_data(const Dataset &full_data, Dataset &local_data) {
    // Calculate the chunks
    const int D = cfg_.D;
    counts_.resize(size_);
    displs_.resize(size_);
    int base = cfg_.N / size_;
    int rem = cfg_.N % size_;
    int offset = 0;
    for (int r = 0; r < size_; ++r) {
      counts_[r] = base + (r < rem ? 1 : 0);
      displs_[r] = offset;
      offset += counts_[r];
    }
    local_N_ = counts_[rank_];

    // If master, send chunks
    vector<double> local_flat;
    if (rank_ == 0) {
      vector<double> full_flat = flatten(full_data, D);
      local_flat.assign(full_flat.begin() + displs_[0] * D,
                        full_flat.begin() + (displs_[0] + counts_[0]) * D);
      for (int dest = 1; dest < size_; ++dest) {
        int n_points = counts_[dest];
        MPI_Send(&n_points, 1, MPI_INT, dest, 0, comm_);
        bytes_sent_local_ += sizeof(int);
        MPI_Send(full_flat.data() + displs_[dest] * D, n_points * D, MPI_DOUBLE,
                 dest, 1, comm_);
        bytes_sent_local_ +=
            static_cast<long long>(n_points) * D * sizeof(double);
      }
      // If worker, receive chunks
    } else {
      MPI_Recv(&local_N_, 1, MPI_INT, 0, 0, comm_, MPI_STATUS_IGNORE);
      bytes_recv_local_ += sizeof(int);
      local_flat.resize(local_N_ * D);
      MPI_Recv(local_flat.data(), local_N_ * D, MPI_DOUBLE, 0, 1, comm_,
               MPI_STATUS_IGNORE);
      bytes_recv_local_ +=
          static_cast<long long>(local_N_) * D * sizeof(double);
    }

    local_data = unflatten(local_flat, local_N_, D);
  }

  // Run Lloyd's algorithm once, instrumenting all MPI calls.
  tuple<vector<int>, vector<double>, int, MPIRunStats>
  run(const Dataset &full_data) {
    const int N = cfg_.N;
    const int D = cfg_.D;
    const int K = cfg_.K;

    MPIRunStats stats;
    bytes_sent_local_ = 0;
    bytes_recv_local_ = 0;

    auto t_run_start = MPI_Wtime();

    // ---- Star scatter -----------------------------------------------------
    auto t0 = MPI_Wtime();
    Dataset local_data;
    scatter_data(full_data, local_data);
    auto t1 = MPI_Wtime();
    stats.scatter_ms = (t1 - t0) * 1000.0;

    // ---- Initialise centroids on master and broadcast ---------------------
    vector<double> centroids_flat(K * D);
    if (rank_ == 0) {
      for (int k = 0; k < K; ++k)
        for (int d = 0; d < D; ++d)
          centroids_flat[k * D + d] = full_data[k][d];
    }

    t0 = MPI_Wtime();
    MPI_Bcast(centroids_flat.data(), K * D, MPI_DOUBLE, 0, comm_);
    t1 = MPI_Wtime();
    double init_bcast_ms = (t1 - t0) * 1000.0;
    long long centroid_bytes = static_cast<long long>(K) * D * sizeof(double);
    if (rank_ == 0)
      bytes_sent_local_ += centroid_bytes;
    else
      bytes_recv_local_ += centroid_bytes;

    // ---- Iterative Lloyd's algorithm --------------------------------------
    vector<int> assignments(local_N_, -1);
    vector<double> local_sums(K * D, 0.0), new_centroids(K * D, 0.0);
    vector<int> local_counts(K, 0), global_counts(K, 0);

    int iter = 0;
    int converged = 0;
    stats.iter_comm_ms = 0.0;

    const long long reduce_sum_bytes =
        static_cast<long long>(K) * D * sizeof(double);
    const long long reduce_count_bytes =
        static_cast<long long>(K) * sizeof(int);

    for (; iter < cfg_.max_iters && !converged; ++iter) {
      // Reset local accumulators
      std::fill(local_sums.begin(), local_sums.end(), 0.0);
      std::fill(local_counts.begin(), local_counts.end(), 0);

      // Assignment + local accumulation (pure computation)
      for (int i = 0; i < local_N_; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_k = -1;
        for (int k = 0; k < K; ++k) {
          double dist = 0.0;
          for (int d = 0; d < D; ++d) {
            double diff = local_data[i][d] - centroids_flat[k * D + d];
            dist += diff * diff;
          }
          if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
          }
        }
        assignments[i] = best_k;
        local_counts[best_k]++;
        for (int d = 0; d < D; ++d) {
          local_sums[best_k * D + d] += local_data[i][d];
        }
      }

      // Reduce partial sums to master
      t0 = MPI_Wtime();
      MPI_Reduce(local_sums.data(), new_centroids.data(), K * D, MPI_DOUBLE,
                 MPI_SUM, 0, comm_);
      t1 = MPI_Wtime();
      stats.iter_comm_ms += (t1 - t0) * 1000.0;
      if (rank_ == 0)
        bytes_recv_local_ += (size_ - 1) * reduce_sum_bytes;
      else
        bytes_sent_local_ += reduce_sum_bytes;

      // Reduce partial counts to master
      t0 = MPI_Wtime();
      MPI_Reduce(local_counts.data(), global_counts.data(), K, MPI_INT, MPI_SUM,
                 0, comm_);
      t1 = MPI_Wtime();
      stats.iter_comm_ms += (t1 - t0) * 1000.0;
      if (rank_ == 0)
        bytes_recv_local_ += (size_ - 1) * reduce_count_bytes;
      else
        bytes_sent_local_ += reduce_count_bytes;

      // Master updates centroids and checks convergence
      if (rank_ == 0) {
        double max_shift = 0.0;
        for (int k = 0; k < K; ++k) {
          if (global_counts[k] == 0) {
            Point p = deterministic_reinit(full_data, N, cfg_.seed, k, iter);
            for (int d = 0; d < D; ++d)
              new_centroids[k * D + d] = p[d];
            global_counts[k] = 1;
          }
          for (int d = 0; d < D; ++d) {
            new_centroids[k * D + d] /= global_counts[k];
            double diff =
                std::abs(new_centroids[k * D + d] - centroids_flat[k * D + d]);
            if (diff > max_shift)
              max_shift = diff;
          }
        }
        converged = (max_shift < cfg_.epsilon) ? 1 : 0;
      }

      // Broadcast convergence decision
      t0 = MPI_Wtime();
      MPI_Bcast(&converged, 1, MPI_INT, 0, comm_);
      t1 = MPI_Wtime();
      stats.iter_comm_ms += (t1 - t0) * 1000.0;
      if (rank_ == 0)
        bytes_sent_local_ += sizeof(int);
      else
        bytes_recv_local_ += sizeof(int);

      // Broadcast updated centroids
      t0 = MPI_Wtime();
      MPI_Bcast(new_centroids.data(), K * D, MPI_DOUBLE, 0, comm_);
      t1 = MPI_Wtime();
      stats.iter_comm_ms += (t1 - t0) * 1000.0;
      if (rank_ == 0)
        bytes_sent_local_ += centroid_bytes;
      else
        bytes_recv_local_ += centroid_bytes;

      centroids_flat.swap(new_centroids);
    }

    // ---- Gather assignments back to master --------------------------------
    vector<int> all_counts, all_displs;
    if (rank_ == 0) {
      all_counts.resize(size_);
      all_displs.resize(size_);
    }

    t0 = MPI_Wtime();
    MPI_Gather(&local_N_, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, comm_);
    t1 = MPI_Wtime();
    stats.gather_ms = (t1 - t0) * 1000.0;
    if (rank_ == 0)
      bytes_recv_local_ += size_ * sizeof(int);
    else
      bytes_sent_local_ += sizeof(int);

    if (rank_ == 0) {
      int off = 0;
      for (int r = 0; r < size_; ++r) {
        all_displs[r] = off;
        off += all_counts[r];
      }
    }

    t0 = MPI_Wtime();
    vector<int> all_assignments;
    if (rank_ == 0)
      all_assignments.resize(N);
    MPI_Gatherv(assignments.data(), local_N_, MPI_INT, all_assignments.data(),
                all_counts.data(), all_displs.data(), MPI_INT, 0, comm_);
    t1 = MPI_Wtime();
    stats.gather_ms += (t1 - t0) * 1000.0;
    if (rank_ == 0)
      bytes_recv_local_ += static_cast<long long>(N) * sizeof(int);
    else
      bytes_sent_local_ += static_cast<long long>(local_N_) * sizeof(int);

    auto t_run_end = MPI_Wtime();
    stats.total_ms = (t_run_end - t_run_start) * 1000.0;
    stats.comm_ms =
        stats.scatter_ms + init_bcast_ms + stats.iter_comm_ms + stats.gather_ms;
    stats.comp_ms = stats.total_ms - stats.comm_ms;
    stats.iters = iter;

    // Aggregate byte counters across all ranks
    MPI_Reduce(&bytes_sent_local_, &stats.bytes_sent, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, comm_);
    MPI_Reduce(&bytes_recv_local_, &stats.bytes_recv, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, comm_);

    return {all_assignments, centroids_flat, iter, stats};
  }
};
#endif // USE_MPI

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
#ifndef USE_MPI
int main() {
  Config cfg;

  std::cout << "=== Sequential K-Means Baseline ===\n";
  std::cout << "N=" << cfg.N << " D=" << cfg.D << " K=" << cfg.K << "\n";

  // 1. Generate data once
  Timer t_gen;
  Dataset data = generate_dataset(cfg);
  double gen_ms = t_gen.elapsed_ms();
  std::cout << "Data generation: " << std::fixed << std::setprecision(2)
            << gen_ms << " ms\n";

  // 2. Run sequential k-means REPEATS times
  vector<double> seq_times;
  vector<Point> final_centroids;
  int final_iters = 0;

  for (int r = 0; r < REPEATS; ++r) {
    Timer t;
    auto [assignments, centroids, iterations] = sequential_kmeans(data, cfg);
    seq_times.push_back(t.elapsed_ms());
    if (r == REPEATS - 1) {
      final_centroids = std::move(centroids);
      final_iters = iterations;
    }
  }

  Stats st = compute_stats(seq_times);

  // Write centroids so the MPI run can compare against them
  write_centroids(final_centroids, "centroids_seq.txt");

  std::cout << "\nSequential K-Means (mean of " << REPEATS
            << " runs): " << std::fixed << std::setprecision(2) << st.mean
            << " ms (+/- " << st.stddev << " ms)\n";
  std::cout << "Iterations: " << final_iters << "\n";
  std::cout << "Final centroids:\n";
  for (int k = 0; k < cfg.K; ++k) {
    std::cout << "  C" << k << ": ";
    for (int d = 0; d < cfg.D; ++d) {
      std::cout << std::setprecision(4) << final_centroids[k][d] << " ";
    }
    std::cout << "\n";
  }

  // Machine-readable line for the justfile to collect
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "CSVSEQ," << st.mean << "," << st.stddev << "," << gen_ms << ","
            << final_iters << "\n";
  return 0;
}
#else
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Config cfg;

  if (rank == 0) {
    std::cout
        << "=== Distributed K-Means (Star / Master-Worker Topology) ===\n";
    std::cout << "MPI processes: " << size << "\n";
    std::cout << "N=" << cfg.N << " D=" << cfg.D << " K=" << cfg.K << "\n\n";
  }

  // Master generates the full dataset once
  Dataset full_data;
  double gen_ms = 0.0;
  if (rank == 0) {
    Timer t_gen;
    full_data = generate_dataset(cfg);
    gen_ms = t_gen.elapsed_ms();
    std::cout << "Data generation (rank 0): " << std::fixed
              << std::setprecision(2) << gen_ms << " ms\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Repeat the distributed run REPEATS times
  vector<double> total_times, comm_times, comp_times;
  vector<int> iters_vec;
  MPIRunStats last_stats;
  vector<double> final_centroids;
  int final_iters = 0;

  MPIStarKMeans solver(cfg, MPI_COMM_WORLD);
  for (int r = 0; r < REPEATS; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    auto [assignments_unused, cents, it, st] = solver.run(full_data);
    double t1 = MPI_Wtime();

    total_times.push_back((t1 - t0) * 1000.0);
    comm_times.push_back(st.comm_ms);
    comp_times.push_back(st.comp_ms);
    iters_vec.push_back(it);

    last_stats = st;
    final_centroids = std::move(cents);
    final_iters = it;
  }

  Stats total_st = compute_stats(total_times);
  Stats comm_st = compute_stats(comm_times);
  Stats comp_st = compute_stats(comp_times);

  if (rank == 0) {
    // Compare against the sequential baseline centroids
    double max_error = -1.0;
    auto seq_centroids = read_centroids("centroids_seq.txt", cfg.K, cfg.D);
    if (seq_centroids.size() == static_cast<size_t>(cfg.K)) {
      max_error = 0.0;
      for (int k = 0; k < cfg.K; ++k) {
        for (int d = 0; d < cfg.D; ++d) {
          max_error =
              std::max(max_error, std::abs(seq_centroids[k][d] -
                                           final_centroids[k * cfg.D + d]));
        }
      }
    }

    std::cout << "\nDistributed K-Means (mean of " << REPEATS
              << " runs): " << std::fixed << std::setprecision(2)
              << total_st.mean << " ms (+/- " << total_st.stddev << " ms)\n";
    std::cout << "  Communication: " << comm_st.mean << " ms (+/- "
              << comm_st.stddev << " ms)\n";
    std::cout << "  Computation:   " << comp_st.mean << " ms (+/- "
              << comp_st.stddev << " ms)\n";
    std::cout << "  Comm/Total ratio: " << std::setprecision(4)
              << (comm_st.mean / total_st.mean) << "\n";
    std::cout << "  Iterations: " << final_iters << "\n";
    std::cout << "  Total bytes sent: " << last_stats.bytes_sent
              << "  received: " << last_stats.bytes_recv << "\n";
    std::cout << "  Max centroid error vs sequential: " << std::scientific
              << max_error << "\n";

    std::cout << "Final centroids:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int k = 0; k < cfg.K; ++k) {
      std::cout << "  C" << k << ": ";
      for (int d = 0; d < cfg.D; ++d) {
        std::cout << final_centroids[k * cfg.D + d] << " ";
      }
      std::cout << "\n";
    }

    std::ostringstream csv;
    csv << std::fixed << std::setprecision(4);
    csv << "CSVMPI," << size << "," << total_st.mean << "," << total_st.stddev
        << "," << gen_ms << "," << final_iters << "," << comm_st.mean << ","
        << comp_st.mean << "," << (comm_st.mean / total_st.mean) << ","
        << last_stats.bytes_sent << "," << last_stats.bytes_recv << ","
        << std::scientific << max_error;
    std::cout << csv.str() << "\n";
  }

  MPI_Finalize();
  return 0;
}
#endif // USE_MPI
