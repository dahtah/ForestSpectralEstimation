#Some functions for computing generalised Markov bounds
#Given a moment sequence s_0, s_1 ..., s_k
#find bounds on μ([a,x])
#for any measure on [a,b] with moment sequence s.
#NB: theoretically this could all be implemented using quadratures but
#the Krein & Nudelman book is inscrutable on the specifics.
#Here we use an approximation of the dual problem that seems to work well enough for low degrees

using JuMP
import Hypatia

#Compute a generalised Chebyshev bound for moment sequence s
#at τ, i.e. bound μ([a,τ]) given moments for prob. measures on [a,b]
#This uses a discrete approximation of the dual constraints with ng grid points
#Increase ng for greater accuracy
function cheb_bound_dual(s,τ;a=-1,b=1,ng=100,debug=false)
    model = Model(() -> Hypatia.Optimizer(verbose = false))
    @assert is_admissible(s,a,b)
    x=range(a,b,ng)
    V=[x^i for x in x, i in 0:(length(s)-1)]
    d = x .< τ
    @variable(model, β[1:length(s)])
    @objective(model, Min, dot(β,s))
    @constraint(model, con, V*β >= d)
    optimize!(model)
    upenv=V*value.(β)
    up= objective_value(model)
    delete(model,con)
    unregister(model,:con)
    @objective(model, Max, dot(β,s))
    @constraint(model,V*β <= d)

    optimize!(model)
    lw= objective_value(model)
    lwenv=V*value.(β)
    (lw,up)
end

