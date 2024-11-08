using LinearAlgebra
using LinearAlgebra: libblastrampoline, BlasInt, require_one_based_indexing
using LinearAlgebra.LAPACK: liblapack, chkstride1, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using BenchmarkTools

include("zlarfg.jl")
include("ztsmqr.jl")
include("axpy.jl")

function lapack_tsqrt!(::Type{T}, l::Int64, A::AbstractMatrix{T}, B::AbstractMatrix{T}, Tau::AbstractMatrix{T}) where{T<: Number}
    m,n = size(B)
    nb = max(1, stride(Tau,2))

    lda = max(1, stride(A,2))
    ldb = max(1, stride(B,2))
    ldt = max(1, stride(Tau,2))

    work = Vector{T}(undef, nb*n)
    info = Ref{BlasInt}(0)

    if T == ComplexF64
        ccall((@blasfunc(ztpqrt_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m,n,l,nb, A, lda, B, ldb, Tau, ldt, work, info)

    elseif T == Float64
        ccall((@blasfunc(dtpqrt_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m,n,l,nb, A, lda, B, ldb, Tau, ldt, work, info)

    elseif T == ComplexF32
        ccall((@blasfunc(ctpqrt_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m,n,l,nb, A, lda, B, ldb, Tau, ldt, work, info)

    else # T = Float32
        ccall((@blasfunc(stpqrt_), libblastrampoline), Cvoid,
        ( Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, 
            Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
            m,n,l,nb, A, lda, B, ldb, Tau, ldt, work, info)
    end
end

function ztsqrt(m, n, ib, A1, lda1, A2, lda2, T, ldt, tau, work)
    # check input Arguments

    if m < 0
        throw(ArgumentError("illegal value of m"))
        return -1
    end

    if n < 0
        throw(ArgumentError("illegal value of n"))
        return -2
    end

    if ib < 0
        throw(ArgumentError("illegal value of ib"))
        return -3
    end

    if lda1 < max(1,m) && m > 0
        throw(ArgumentError("illegal value of lda1"))
        return -5
    end

    if lda2 < max(1,m) && m > 0
        throw(ArgumentError("illegal value of lda2"))
        return -7
    end

    if ldt < max(1,ib) && ib > 0
        throw(ArgumentError("illegal value of ldt"))
        return -9
    end

    # quick return 
    if m == 0 || n == 0 || ib == 0
        return
    end

    one0 = oneunit(eltype(A1))
    zero0 = zero(eltype(A1))
    plus = LinearAlgebra.MulAddMul(one0, one0)

    for ii in 1:ib:n
        sb = min(n-ii+1, ib)

        for i in 1:sb
            # generate elementary reflector H[ii*ib + i] to annilate A[ii*ib, + i:m, ii*ib + i]
            A1[ii+i-1, ii+i-1], tau[ii+i-1] = zlarfg(m+1, A1[ii+i-1, ii+i-1], (@view A2[1:m, ii+i-1]), 1, tau[ii+i-1])

            if ii+i <= n
                # apply H[ii*ib + i] to A[ii*ib + i:m, ii*ib + i + 1 : ii*ib + ib] from left
                alpha = -conj(tau[ii+i-1])
                (@view work[1:sb-i]) .= (@view A1[ii+i-1, ii+i:ii+sb-1])
                
                conj!((@view work[1:sb-i]))
                LinearAlgebra.generic_matvecmul!((@view work[1:sb-i]), 'C', (@view A2[1:m, ii+i:ii+sb-1]), (@view A2[1:m, ii+i-1]), plus)
                conj!((@view work[1:sb-i]))
                LinearAlgebra.axpy!(alpha, (@view work[1:sb-i]), (@view A1[ii+i-1, ii+i:ii+sb-1]))
                conj!((@view work[1:sb-i]))
                gerc!(alpha, (@view A2[1:m, ii+i-1]), (@view work[1:sb-i]), (@view A2[1:m, ii+i:ii+sb-1]))
            end

            # Calculate T
            alpha = -tau[ii+i-1]
            LinearAlgebra.generic_matvecmul!((@view T[1:i-1, ii+i-1]), 'C', (@view A2[1:m, ii:ii+i-2]), (@view A2[1:m, ii+i-1]),LinearAlgebra.MulAddMul(alpha, zero0))
            #LinearAlgebra.BLAS.trmv!('U', 'N', 'N', (@view T[1:i-1, ii:ii+i-2]), (@view T[1:i-1, ii+i-1]))
            LinearAlgebra.generic_trimatmul!((@view T[1:i-1, ii+i-1]), 'U', 'N', identity, (@view T[1:i-1, ii:ii+i-2]), (@view T[1:i-1, ii+i-1]))
            T[i, ii+i-1] = tau[ii+i-1]
        end

        if n >= ii+sb
            ww = reshape(@view(work[1: ib*(n-(ii+sb)+1)]), ib, n-(ii+sb)+1)

            ztsmqr('L', 'C', sb, n-(ii+sb) + 1, m, n-(ii+sb) + 1, ib, ib, 
            (@view A1[ii:ii+sb-1, ii+sb: n]), sb, (@view A2[1:m, ii+sb:n]), m, 
            (@view A2[1:m, ii:ii+sb-1]), m, (@view T[1:ib, ii:ii+ib-1]), ib, ww, sb)
        end
    end
end
