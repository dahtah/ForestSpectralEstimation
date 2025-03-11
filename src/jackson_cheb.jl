#Some code for Jackson-Chebyshev estimation of the spectral cdf
#TODO: document!
function estimate_cdf_JCpoly(g::AbstractGraph, λ_grid::AbstractVector, N_rv::Integer, trunc_ind::Integer, method=:jackson_cheby)
    N_λ = length(λ_grid)
    λmax = λ_grid[end]
    L = laplacian_matrix(g)
    n = nv(g)

    CH = zeros(N_λ, trunc_ind)
    for i = 1:N_λ
        CH[i,:] = cheby_poly_coefficients_of_step_function(λ_grid[i], λmax, trunc_ind, method)
    end
    random_matrix = randn(n, N_rv) ./ sqrt(n) # create N_rv random Gaussian vectors of size n and identity covariance matrix

    Tk_est = est_k_cheby(L, trunc_ind, random_matrix, λmax)
    #Tk = k_cheby_exact(L, trunc_ind, λmax) ./ n; plot(Tk); plot!(Tk_est);  #without random vectors

    return vec(sum(CH' .* Tk_est, dims=1)), CH
end

function estimate_cdf_JCpoly_IS(g::AbstractGraph, λ_grid::AbstractVector, N_rv::Integer, N_IS::Integer, trunc_ind::Integer, method)
    N_λ = length(λ_grid)
    λmax = λ_grid[end]
    L = laplacian_matrix(g)
    n = nv(g)

    CH = zeros(N_λ, trunc_ind)
    for i = 1:N_λ
        CH[i,:] = cheby_poly_coefficients_of_step_function(λ_grid[i], λmax, trunc_ind, method)
    end
    random_matrix = randn(n, N_rv) ./ sqrt(n) # create N_rv random Gaussian vectors of size n and identity covariance matrix

    Tk_est = est_k_cheby(L, trunc_ind, random_matrix, λmax)
    #Tk_est_cv = est_k_cheby_cv(L, trunc_ind, random_matrix, λmax)
    #Tk = k_cheby_exact(L, trunc_ind, λmax) ./ n; plot(Tk); plot!(Tk_est); plot!(Tk_est_cv) #without random vectors

    est_cdf = zeros(N_λ)
    for i = 1:N_λ
        aCH = abs.(CH[i,:])
        Zi = sum(aCH)
        wv = Weights(aCH ./ Zi)
        counts_IS = zeros(N_IS)
        for j=1:N_IS
            k = sample(wv)
            counts_IS[j] = sign(CH[i,k]) * Tk_est[k] # the true IS estimator is (CH[i,k] / abs.(CH[i,:])) * Zi *  Tk_est[k]. The Zi multiplication can however be done after this for loop
        end
        est_cdf[i] = mean(counts_IS) * Zi
    end
    return est_cdf, CH
end


function est_k_cheby(L :: AbstractMatrix, trunc_ind:: Integer, random_matrix::AbstractMatrix, λmax)

    n = size(L,1)
    N_rv = size(random_matrix,2)

    @assert size(random_matrix,1) == n
    @assert λmax > 0
    @assert trunc_ind > 2

    sq_sum = zeros(trunc_ind, N_rv)

    A = (2/λmax) * L - spdiagm(0 => ones(n))

    for i = 1:N_rv
        x = random_matrix[:,i]
        T_km2 = x
        sq_sum[1,i] = norm(x)^2
        T_km1 = A * T_km2
        sq_sum[2,i] = x' * T_km1
        for l = 3:trunc_ind
            T_new = 2 * A * T_km1 - T_km2
            sq_sum[l,i] = x' * T_new
            T_km2 = T_km1
            T_km1 = T_new
        end
    end
    return vec(mean(sq_sum, dims=2))
end

"""
This function computes the m+1 first coefficients of the chebychev expansion of 
the ideal low pass function with threshold b (i.e., the function f(x) defined on [0, λmax]
that returns 1 if x < c and 0 if not).

Usage
----------
```ch = cheby_poly_coefficients_of_step_function(b, λmax, m)```

Entry
----------
* ```c``` : the cut-off of the filter. Must verify ```0 < c < λmax``` (Float64)
* ```λmax``` : the range up to which one whishes the approximation. Must be positive (Float64)
* ```m``` : the order to which we compute the coefficients (Int64)

Returns
-------
```ch``` : returns a vector of m+1 polynomial coefficients.
"""
function cheby_poly_coefficients_of_step_function(b, λmax, m::Integer, method)
    if method == :cheby
        @assert 0 < b <= λmax
        # when we map [0, λmax] to [-1,1], the cut-off becomes:
        b = (2 * b / λmax) - 1
        acosb = acos(b)
        CH = zeros(m)
        CH[1] = 1 - ( acosb / pi )
        for k=1:m-1
            CH[k+1] = - 2 * sin(k * acosb) / (k * pi)
        end
    elseif method == :jackson_cheby
        CH = cheby_poly_coefficients_of_step_function(b, λmax, m, :cheby)
        gamma_JACK = zeros(m)
        alpha = pi/(m+1)
        gamma_JACK[1] = 1
        for k=1:m-1
            gamma_JACK[k+1] = (1/sin(alpha))*((1-k/(m+1))*sin(alpha)*cos(k*alpha)+(1/(m+1))*cos(alpha)*sin(k*alpha))
        end
        CH = CH .* gamma_JACK
    elseif method == :von_mises # this is NOT working
        κ = 5000 #as chosen in the paper
        @assert 0 < b <= λmax
        # when we map this problem to [-1,1], the cut-off becomes:
        b = (2 * b / λmax) - 1
        acosb = acos(b)
        CH = zeros(m)
        CH[1] = 2 / pi
        for k=1:m-1
            CH[k+1] = (2/pi) * exp(-0.5 * (k^2)/κ) * cos(k * acosb)
        end
    else
        error("Unknown method")
    end
    return CH
end




############################################################### more functions ###############################################################"

function cheby_eval(x, CH, λmax)
    n = length(x)
    A = (2/λmax) * spdiagm(0 => x) - spdiagm(0 => ones(n))

    T_km2 = ones(n)
    T_km1 = A * T_km2
    r = CH[1] * T_km2 + CH[2] * T_km1
    for k = 2:length(CH)-1
        T_new = 2 * A * T_km1 - T_km2
        r += CH[k+1] * T_new
        T_km2 = T_km1
        T_km1 = T_new
    end
    return r
end


# Function to create a Kneser graph K(n, k)
function kneser_graph(n, k)
    # Generate all k-element subsets of a set of size n
    vertices = collect(combinations(1:n, k))
    
    # Create an empty graph
    g = Graph(length(vertices))
    
    # Compare each pair of subsets (vertices) for disjointness
    for i in eachindex(vertices)
        for j in i+1:length(vertices)
            # Check if the two subsets are disjoint
            if isempty(intersect(vertices[i], vertices[j]))
                add_edge!(g, i, j)
            end
        end
    end
    return g
end




function k_cheby_exact(L, trunc_ind::Int64, λmax::Float64)

    n = size(L,1)
    @assert λmax > 0
    @assert trunc_ind > 2

    tr_k_cheby = zeros(trunc_ind)

    A = (2/λmax) * L - spdiagm(0 => ones(n))

    T_km2 = spdiagm(0 => ones(n))
    tr_k_cheby[1] = tr(T_km2)
    T_km1 = A
    tr_k_cheby[2] = tr(T_km1)
    T_new = 2 * A * T_km1 - T_km2
    for l = 3:trunc_ind
        T_new = 2 * A * T_km1 - T_km2
        tr_k_cheby[l] = tr(T_new)
        T_km2 = T_km1
        T_km1 = T_new
    end

    return tr_k_cheby
end
