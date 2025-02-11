
using Plots,Graphs,LinearAlgebra,Statistics
import ForestSpectralEstimation as FSE
import Combinatorics


# Function to create a Kneser graph K(n, k)
function kneser_graph(n, k)
    # Generate all k-element subsets of a set of size n
    vertices = collect(Combinatorics.combinations(1:n, k))
    
    # Create an empty graph
    g = Graph(length(vertices))
    
    # Compare each pair of subsets (vertices) for disjointness
    for i in 1:length(vertices)
        for j in i+1:length(vertices)
            # Check if the two subsets are disjoint
            if isempty(intersect(vertices[i], vertices[j]))
                add_edge!(g, i, j)
            end
        end
    end
    return g
end



function demo(g;exact=:false,nqs=30,nm=4,nrep=10,n_max_exact=5000,method=:truncate)
    #Default range is logarithmic on .05*d_bar to 4*dbar
    qs = FSE.default_range(g,nqs)

    #Compute moments using independent forests for all values of q in qs
    #nm: number of moments (default 4)
    #nrep: number of repetitions 
    res=FSE.collect_moments(g,qs=qs,nm=nm,nrep=nrep)

    #Run the different CDF reconstruction algorithms on the set of collected moments
    out=FSE.reconstruct(res.qs,res.moments,g,method=method)
    p=plot(out.qs,[out.lw out.up out.maxent],labels=["Lw." "Up." "ME"])
    plot!(FSE.isotonic(out.qs,out.maxent)...,labels="ME (corr)")
    if exact && (nv(g) < n_max_exact)
        if nv(g) > n_max_exact
            @info "Graph is too large, ignoring request for exact EVD"
        end
        l=eigvals(Symmetric(Matrix(laplacian_matrix(g))))
        plot!(out.qs,[mean(l .< q) for q in out.qs],label="Exact",linestyle=:dot,linewidth=2)
    end
    p
end
