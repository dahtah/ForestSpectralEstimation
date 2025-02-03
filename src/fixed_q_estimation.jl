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

function chebbounds_fixedq(y,a,b)
    #This returns a Chebyshev-Krein bound for the cdf at 0.5
    cb = cheb_bound_dual([1;y],.5,a=a,b=b)
    lw=1-cb[2] #we need p(z >= 1/2), reverse
    up=1-cb[1]
    (lw,up)
end


function maxent_fixedq(y,v,a,b)
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
        res=maxent_confidence(ef,y,sqrt.(v[1:m]))
        return (prop=expectation(res.ed,(v)-> itr(v) > 0.5),ed=res.ed,status=:success)
    catch #Optim. failed
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


#Given a sequence of moments at q_1 ... q_m
#try to reconstruct the cdf 
function reconstruct(qs,moments,g)
    me = zeros(length(qs)) 
    lw = zeros(length(qs))
    up = zeros(length(qs))
    for i in 1:length(qs)
        q = qs[i]
        a,b=range_q(g,q)
        #Truncate the moments
        y,v = moments[i]
        yt=truncate_moments_alex(y,v,a=a,b=b)
        m = length(yt) #number of remaining moments
        lw[i],up[i]=chebbounds_fixedq(yt,a,b)
        me[i] = maxent_fixedq(yt,v[1:m],a,b).prop
    end
    (qs=qs,maxent=me,lw=lw,up=up)
end
