set shell := ["bash", "-c"]

# Part 1: OpenMP matrix multiplication benchmark
one:
    cd "Part 1" && \
    clang++ -std=c++17 -O2 -Xpreprocessor -fopenmp \
        -I/opt/homebrew/opt/libomp/include \
        -L/opt/homebrew/opt/libomp/lib \
        -lomp solution.cpp -o solution
    cd "Part 1" && ./solution

# Part 2: sequential K-Means baseline (no MPI)
two:
    cd "Part 2" && clang++ -std=c++17 -O2 solution.cpp -o solution
    cd "Part 2" && ./solution

# Part 2: MPI star-topology K-Means only (requires mpicxx / mpirun)
mpi:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "Part 2"
    # Force shared-memory transport only - avoids macOS firewall pop-ups
    export OMPI_MCA_btl=self,sm
    mpicxx -std=c++17 -O2 -DUSE_MPI solution.cpp -o solution_mpi
    for np in 2 4 6 8; do
        mpirun --oversubscribe -np ${np} ./solution_mpi > /tmp/part2_mpi_${np}.txt
        tail -n 1 /tmp/part2_mpi_${np}.txt
    done

# Part 2: full benchmark - sequential baseline + MPI runs, writes performance_part2.csv
part2:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "Part 2"
    # Force shared-memory transport only - avoids macOS firewall pop-ups
    export OMPI_MCA_btl=self,sm

    # 1. Sequential baseline
    clang++ -std=c++17 -O2 solution.cpp -o solution
    ./solution | tee /tmp/part2_seq.txt

    seq_mean=$(grep '^CSVSEQ,' /tmp/part2_seq.txt | cut -d',' -f2)
    seq_std=$(grep '^CSVSEQ,' /tmp/part2_seq.txt | cut -d',' -f3)
    seq_gen=$(grep '^CSVSEQ,' /tmp/part2_seq.txt | cut -d',' -f4)
    seq_iters=$(grep '^CSVSEQ,' /tmp/part2_seq.txt | cut -d',' -f5)

    # 2. Build MPI executable
    mpicxx -std=c++17 -O2 -DUSE_MPI solution.cpp -o solution_mpi

    # 3. Write CSV header and baseline row
    echo "Processes,Version,MeanTime_ms,StdDevTime_ms,DataGen_ms,Iterations,CommTime_ms,CompTime_ms,CommRatio,BytesSent,BytesRecv,MaxCentroidError,Speedup" > performance_part2.csv
    echo "1,Sequential,${seq_mean},${seq_std},${seq_gen},${seq_iters},0,0,0,0,0,0,1.0" >> performance_part2.csv

    # 4. Run MPI for 2, 4, 6, 8 processes and append rows
    for np in 2 4 6 8; do
        mpirun --oversubscribe -np ${np} ./solution_mpi > /tmp/part2_mpi_${np}.txt
        line=$(grep '^CSVMPI,' /tmp/part2_mpi_${np}.txt)
        mpi_mean=$(echo "$line" | cut -d',' -f3)
        mpi_std=$(echo "$line" | cut -d',' -f4)
        mpi_gen=$(echo "$line" | cut -d',' -f5)
        mpi_iters=$(echo "$line" | cut -d',' -f6)
        mpi_comm=$(echo "$line" | cut -d',' -f7)
        mpi_comp=$(echo "$line" | cut -d',' -f8)
        mpi_ratio=$(echo "$line" | cut -d',' -f9)
        mpi_sent=$(echo "$line" | cut -d',' -f10)
        mpi_recv=$(echo "$line" | cut -d',' -f11)
        mpi_err=$(echo "$line" | cut -d',' -f12)
        speedup=$(awk -v s="${seq_mean}" -v m="${mpi_mean}" 'BEGIN{printf "%.2f", s/m}')
        echo "${np},MPI_Star,${mpi_mean},${mpi_std},${mpi_gen},${mpi_iters},${mpi_comm},${mpi_comp},${mpi_ratio},${mpi_sent},${mpi_recv},${mpi_err},${speedup}" >> performance_part2.csv
    done

    echo ""
    echo "CSV written to: Part 2/performance_part2.csv"
    cat performance_part2.csv
