
#Basic Lanczos iteration
#No attempt is made to ensure numerical stability (ie. no reorthogonalisation)
#Should only be used for small k
function lanczos(A, q, k)
    n = size(A, 1)  # Size of the matrix A
    Q = zeros(n, k)  # Orthogonal basis vectors
    α = zeros(k)     # Diagonal entries of T
    β = zeros(k-1)   # Off-diagonal entries of T

    # Initialize the first vector
    q = q / norm(q)  # Normalize
    Q[:, 1] = q

    # First iteration
    v = A * q
    α[1] = dot(q, v)
    v = v - α[1] * q

    for i in 2:k
        β[i-1] = norm(v)
        if β[i-1] ≈ 0 
            break
        end
        q = v / β[i-1]
        Q[:, i] = q

        v = A * q
        α[i] = dot(q, v)
        v = v - α[i] * q - β[i-1] * Q[:, i-1]
    end

    # Construct the tridiagonal matrix T
    return (T=SymTridiagonal(α, β),Q=Q)
end

function lanczos_quad(A,z,k)
    eig = eigen(lanczos(A,z,k).T)
    (x=eig.values,w=eig.vectors[1,:].^2)
end






@doc raw"""
    slq(A,k,nrep)

Stochastic Lanczos quadrature. Returns an approximation of the spectral CDF of matrix $A$, of degree $k$, using nrep replicates. 

# Arguments
- A: any AbstractMatrix
- k: degree of quadrature. Higher degree = less bias, slower. Numerical issues will appear if k is too large.
- nrep: number of replicates (more replicates = less variance, slower)

# Return value

This function returns an estimate of the cdf as an "StatsBase.ECDF" value, which is callable.

# Example

```
D = Diagonal([1,.8,.5,.2])
cf=FSE.slq(D,4,10);
cf(.1) #should be 0
cf(.9) #should be close to 0.75
```

# References

Chen et al., Analysis of stochastic Lanczos quadrature for spectrum approximation; https://arxiv.org/pdf/2105.06595
"""
function slq(A,k,nrep)
    n = size(A,1)
    res = [lanczos_quad(A,normalize(randn(n)),k) for _ in 1:nrep]

    x = reduce(vcat,first.(res))
    w = reduce(vcat,last.(res))
    StatsBase.ecdf(x,weights=StatsBase.weights(w))
end
