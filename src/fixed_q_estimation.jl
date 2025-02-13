#Implements some functions for fixed-q spectral estimation

#Generate nm forests, then estimate all moments using self roots up to
#order nm
#Unoptimised!
#TODO: variance reduction, current estimate is not permutation invariant
function all_forest_moments(g,q,nm=4)
    n = nv(g)
    fs = [random_forest(g,q) for _ in 1:nm]
    y=[self_roots(fs[1:i]) for i in 1:nm] / n
    return y
end

#Repeat moment estimator nrep times at fixed q, return mean and variance estimator
function estimate_forest_moments(g,q;nm=4,nrep=10)
    y = mean((all_forest_moments(g,q,nm) for _ in 1:nrep))
    #Compute an approximate bound on the variance (Alex's bound)
    var = (1/nrep)*(1/(nv(g)))*y
    (y=y,var=var)
end

#Default bound on λ_max on a graph 
function bound_lambda(g)
    2*maximum(degree(g))
end



function qmap(l,q)
    q / (q+l)
end

#For fixed q, q/(q+λ) will be contained in the following interval
function range_q(g,q)
    (q/(q+bound_lambda(g)),1.0)
end

function bounds_fixedq(y,a,b)
    #This returns a Markov-Krein bound for the cdf at 0.5
    cb = markov_bound([1;y],.5,a,b)
    lw=1-cb[2] #we need p(z >= 1/2), reverse
    up=1-cb[1]
    (lw,up)
end


function maxent_fixedq(y,v,a,b;η0=zeros(length(y)))
    @assert length(η0) == length(y)
    @assert length(v) == length(y)

    #For compatibility with the ExpFamily struct,
    #we need to map our variable from the interval [a,b]
    #to the interval [-1,1]
    #FIXME: handle this in ExpFamily directly
    m = length(y)
    tr = (v)-> 2*(v-a)/(b-a) - 1 
    itr = (x)-> .5*(b-a)*(x+1) + a
    ef = ExpFamily((v)->[itr(v)^i for i in 1:m])
    #Find max. entropy estimates 
    try
        res=maxent_confidence(ef,y,sqrt.(v),η0=η0)
        return (prop=expectation(res.ed,(v)-> itr(v) > 0.5),ed=res.ed,status=:success)
    catch err
        return (prop=NaN,ef=ef,status=:failure)  
    end
end


#Alex's heuristic for keeping a subset of moments such that the confidence set
#is entirely contained within the space of moments for measures on the interval
#[a,b]
function truncate_moments_alex(y,var;a=-1.0,b=1.0)
    s = [1.0;y] #augment with trivial moment
    std_err = [0.0;sqrt.(var)]
    s = OffsetArrays.Origin(0)(s)
    std_err = OffsetArrays.Origin(0)(std_err)
    n = length(y) #max degree
    k = 1;
    for i = 1:(n-1)
        lwb,upb=conditional_moment_bounds(s[0:i],a,b)
        lw=s[i+1]-std_err[i+1]
        up=s[i+1]+std_err[i+1]
        if (lw < lwb) || (up > upb)
            break
        else
            k+=1
        end
    end
    y[1:k]
end

function default_range(g,ns=30)
    db=mean(degree(g))
    exp.(range(log(.05*db),log(4*db),ns))
end

function collect_moments(g;qs=default_range(g),nm=4,nrep=10)
    moments = [estimate_forest_moments(g,q,nm=nm,nrep=nrep) for q in qs]
    (qs=qs,moments=moments)
end

#Denoise and truncate if denoised moments are too far from empirical mean
function adaptive_denoising(y,v;a=-1.,b=1.)
    n = length(y)
    w = [1; 1 ./ v] #Weigths to use in weighted LS denoising
    if is_admissible([1;y[1:n]],a,b)
        return y
    end
    while n > 1
        yt = denoise([1;y[1:n]],a=a,b=b,w=w[1:(n+1)])[2:end]
        res = (yt - y[1:n]) ./ sqrt.(v[1:n])
        if (all(abs.(res) .< 2)) && is_admissible([1;yt],a,b)
            return yt
        else
            n = n-1
        end
    end
    return y[1]
end

#Given a sequence of moments at q_1 ... q_m
#try to reconstruct the cdf
#Returns: me Max. entropy estimate
#lw,up Markov bounds
#nm Number of moments used at each q_i
function reconstruct(qs,moments,g;method=:truncate,warm_start=false)
    me = zeros(length(qs)) 
    lw = zeros(length(qs))
    up = zeros(length(qs))
    nm = zeros(length(qs)) #number of moments used
    #This is used for warm-starting the max. entropy optim.
    η0 = zeros(maximum([length(m.y) for m in moments]))
    for i in 1:length(qs)
        q = qs[i]
        a,b=range_q(g,q)
        y,v = moments[i]
        if method == :project #denoise naively, then keep admissible subset
            yn=denoise([1;y],a=a,b=b)
            yt=admissible_subset(yn,a,b)[2][2:end]
        elseif method == :adaptive
            #denoise, taking measurement variance into account
            yt=adaptive_denoising(y,v,a=a,b=b)
        elseif method == :truncate
            #Alex's method
            yt=truncate_moments_alex(y,v,a=a,b=b)
        end
        m = length(yt)
        nm[i] = m
        lw[i],up[i]=bounds_fixedq(yt,a,b)
        δ= up[i]-lw[i]
        if δ > 1e-3
            res_me = maxent_fixedq(yt,v[1:m],a,b;η0=η0[1:m])
            me[i] = res_me.prop
            if isfinite(me[i]) && warm_start
                #η0[1:3] .= res_me.ed.β[1:3]
                η0[1:m] .= res_me.ed.β[1:m]
                # η0[(m+1):end] .= 0
            end
        else
            me[i] = (up[i]+lw[i])/2
        end
    end
    (qs=qs,maxent=me,lw=lw,up=up,nm=nm)
end
