using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots
using CSV, Tables

include("unified_rec.jl")  # Include unified_rectrxm.jl file

function trsm_flops(t, m, n)
    flops_add = 0.5 * n * m * (m-1.0)
    flops_mult = 0.5 * n * m * (m+1.0)
    return flops_add + flops_mult
end

function benchmark_rectrxm(T)
    Timing = zeros(10)
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    m_values = [128, 256, 1024]  # Different values of m for benchmarking
   
    Timing = zeros(10)

    for m in m_values
        for n in sizes
            Timing[1] = 1;
            Timing[2] = Threads.nthreads();
            Timing[3] = n;
            Timing[4] = m;
            Timing[5] = 256;
            Timing[6] = trsm_flops(T, n, m)

            A = CuArray(Matrix(LowerTriangular(rand(n, n))))
            B = CuArray(Matrix(rand(n, m)))

            Ac = copy(A)
            Bc = copy(B)

            time_rectrxm = @belapsed (CUDA.@sync unified_rectrxm!('L', 'L', 'N', 1.0, 'M', $A, $B))
            Timing[7] = time_rectrxm
            recgflopss = (trsm_flops(T, n, m)/10^9) / time_rectrxm
            Timing[8] = recgflopss
            println("unified_rectrxm! - Size: $n x $m | Runtime: $time_rectrxm s Gflops/s: $recgflopss")

            time_trsm = @belapsed (CUDA.@sync CUDA.CUBLAS.trmm!('L', 'L', 'N', 'N', 1.0, $Ac, $Bc))
            Timing[9] = time_trsm
            cugflopss = (trsm_flops(T, n, m)/10^9) / time_trsm
            Timing[10] = cugflopss
            println("cuBLAS trmm - Size: $n x $m | Runtime: $time_trsm s Gflops/s: $cugflopss")
            CSV.write("timings_trxm_Bnonsquare_CUDA_$(T).csv", Tables.table(transpose(Timing)), writeheader=false, append=true)
        end
    end
end

# Run the benchmark
for T in [Float32, Float64]
    benchmark_rectrxm(T)
end