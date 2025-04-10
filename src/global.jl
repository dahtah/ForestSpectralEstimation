import Clarabel

#Find a discrete weight vector w that approximately solves Aw = b
#Specifically: given A,b, and measurement error δ
#solve argmin Pen(w)
#st (A*w - b) ./ δ <= γ (small residuals)
#where Pen(w) is one of the following penalties
#:l1 => penalises ∑|w_{i+1}-w_{i}|
#:l2 => penalises ∑|w_{i+1}-w_{i}|^2
#:maxent => penalises ∑ w_i log w_i
#The algorithm starts with γ=γ0 and decreases it until the optimisation algorithm fails
#This may be because there is no feasible w for this value of γ or because the conditioning is too poor. 
#Note that δ should be a vector of length b, approximating the std. error of the measurement
#The function returns a tuple (ws,γs,resd) where
#ws[i] is the solution for γs[i], with residual (Aw-b) resd[i]
#Optimisation is performed using JuMP.jl and the Clarabel solver
function reconstruct_density(A,b,δ;γ0,maxit=100,α=.9,penalty=:l1)
    m,n =size(A)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, w[1:n],lower_bound=0)
    @constraint(model, sum(w) == 1)


    if penalty in [:l1,:l2]
        @variable(model, dw[1:(n-1)])
        @variable(model,t)
        for i in 2:n
            @constraint(model, w[i]==dw[i-1]+w[i-1])
        end
        if penalty == :l1
            @constraint(model, [t; dw] in MOI.NormOneCone(1 + length(dw)))
        elseif penalty == :l2
            @constraint(model, [t; dw] in MOI.SecondOrderCone(1 + length(dw)))
        end
        
        @objective(model, Min, t)
    elseif penalty == :maxent

        @variable(model, t[1:n])

        @constraint(model, [i = 1:n], [t[i], w[i], 1] in MOI.ExponentialCone())
        @objective(model, Max, sum(t))
    end

    @variable(model,τ)
    #Constrain residuals
    @constraint(model,[τ;(A*w-b) ./ δ] in MOI.SecondOrderCone(m+1))
    @constraint(model,con,τ==1)
    γs = Vector{Float64}();
    ws=Vector{Vector{Float64}}()
    resd=Vector{Vector{Float64}}()
    γ=γ0
    it=1
    #Try to decrease γ until problem becomes unfeasible
    while it <= maxit
        set_normalized_rhs(con, γ)
        optimize!(model)
        push!(γs,γ)
        push!(ws,value.(w))
        push!(resd,value.(A*w-b))
        if !is_solved_and_feasible(model) #Model isn't solvable anymore
            break
        end
        it += 1
        γ=α*γ
    end
    #assert_is_solved_and_feasible(model)
    (ws=ws,γs=γs,resd=resd)
end


#Generate a constraint matrix for a discretised version of the
#rational moment problem for moments of the form (q/(q+x))^k
function gen_constraint_matrix(g,qs;nm=4,ng=200,lb=1e-5,wlog=false)
    nq=length(qs)
    if wlog
        lx = range(log(lb),log(2*maximum(degree(g))),ng)
        x = exp.(lx)
    else
        x = collect(range(0,2*maximum(degree(g)),ng))
    end
    A = [(q/(q+x))^k for x in x, q in qs, k in 1:nm]
    (x=x,qs=qs,A=reshape(A,ng,nm*nq))
end

#remove the zero eigenvalue from the moments
function remove_l0(y,n)
    (n/(n-1))*(y - 1/n)
end



#Reconstruct the spectral density from a vector of (rational) moments
#using a global strategy.
#penalty
#:l1 => Reconstructed ecdf is piecewise linear
#:l2 => Reconstructed ecdf is twice differentiable
#:maxent => Max entropy prior on density
function reconstruct_global(qs,moments,g;η=10,nq=20,penalty=:l2)
    nm = length(moments[1].y)
    bs=gen_constraint_matrix(g,qs,ng=200,nm=nm)
    be=reduce(vcat,eachrow(reduce(hcat,[m.y for m in moments])))
    be=remove_l0.(be,nv(g)) #Remove the contribution of λ0 (the null eigenvalue) from moments
    err=reduce(vcat,eachrow(reduce(hcat,[sqrt.(m.var) for m in moments])))

    res=reconstruct_density(bs.A',be,err,γ0=η,maxit=120,α=.8,penalty=penalty)
    ms = [dot(bs.x,we) for we in res[1]] #Compute E(λ) for all estimated distributions in regularisation path
    ml=((nv(g))/(nv(g)-1))*mean(Graphs.degree(g)) #Find the actual value 
    iopt=argmin(abs.(ms .- ml)) #Pick the estimate that best predicts the actual value 
    w_opt=res.ws[iopt]
    resd_opt=res.resd[iopt]
    @info "Max. residual $(maximum(abs.(resd_opt)))"
    #Return estimated ECDF
    (ecdf=v-> sum(w_opt[bs.x .<= v]),x=bs.x,w=w_opt)
end


