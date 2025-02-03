
using Plots,Graphs,LinearAlgebra,Statistics
import ForestSpectralEstimation as FSE




function demo(g;exact=:false,nqs=30,nm=4,nrep=10,n_max_exact=5000)
    #Default range is logarithmic on .05*d_bar to 4*dbar
    qs = FSE.default_range(g,nqs)

    #Compute moments using independent forests for all values of q in qs
    #nm: number of moments (default 4)
    #nrep: number of repetitions 
    res=FSE.collect_moments(g,qs=qs,nm=nm,nrep=nrep)

    #Run the different CDF reconstruction algorithms on the set of collected moments
    out=FSE.reconstruct(res.qs,res.moments,g)
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
