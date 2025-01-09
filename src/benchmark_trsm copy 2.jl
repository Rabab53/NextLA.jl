using LinearAlgebra
using CUDA
using BenchmarkTools
using Plots

include("performant_trsm_2 copy.jl")  # Include performant_trsm_2_copy.jl file

function benchmark_trsm()
    # sizes = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
    #          544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]
    sizes = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

    # m_values = [1, 2, 4, 8, 16, 32, 64, 128]  # Different numbers of columns in B
    m_values = [256]

    results = Dict()

    for m in m_values
        trsm_2_copy_runtimes = Float64[]
        cuda_trsm_runtimes = Float64[]

        for n in sizes
            # Generate random lower triangular matrix A and random matrix B
            A = CuArray(Matrix(LowerTriangular(rand(n, n))))  # Lower triangular matrix
            B = CuArray(Matrix(rand(n, m)))  # Matrix B of size n x m

            Ac = copy(A)  # Copy of A for cuBLAS trsm
            Bc = copy(B)  # Copy of B for cuBLAS trsm

            # Benchmark for performant_trsm_2_2! using CUDA.@sync
            time_trsm_2_copy = @belapsed (CUDA.@sync performant_trsm_2_2!('L', 'L', 'N', $A, $B))  # Synchronize GPU timing
            median_runtime_trsm_2_copy = time_trsm_2_copy  # Time in seconds (already synced)
            push!(trsm_2_copy_runtimes, median_runtime_trsm_2_copy)
            println("performant_trsm_2_2! - Size: $n x $n, m: $m | Runtime: $median_runtime_trsm_2_copy s")

            # Benchmark for cuBLAS trsm using CUDA.@sync
            time_cuda_trsm = @belapsed (CUDA.@sync CUDA.CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, $Ac, $Bc))  # Synchronize GPU timing
            median_runtime_cuda_trsm = time_cuda_trsm  # Time in seconds (already synced)
            push!(cuda_trsm_runtimes, median_runtime_cuda_trsm)
            println("cuBLAS trsm - Size: $n x $n, m: $m | Runtime: $median_runtime_cuda_trsm s")
        end

        results[m] = (trsm_2_copy_runtimes, cuda_trsm_runtimes)
    end

    return sizes, results
end

# Run the benchmark
sizes, results = benchmark_trsm()

# Create plots for each m value
for m in keys(results)
    trsm_2_copy_runtimes, cuda_trsm_runtimes = results[m]
    
    plot(
        sizes,
        trsm_2_copy_runtimes,
        label = "performant_trsm_2_2!",
        xlabel = "Matrix Size (n x n)",
        ylabel = "Runtime (s)",
        title = "Runtime Comparison for B with m=$m",
        lw = 2,
        marker = :square,
        markersize = 4,
        color = :green,
        legend = :topleft,
        yscale = :log10  # Log scale to better visualize the differences
    )
    plot!(
        sizes,
        cuda_trsm_runtimes,
        label = "cuBLAS trsm",
        lw = 2,
        marker = :diamond,
        markersize = 4,
        color = :red
    )

    # Save the plot
    savefig("trsm_comparison_m_2_$m.png")
end
