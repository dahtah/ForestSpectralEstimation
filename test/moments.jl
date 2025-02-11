import OffsetArrays

function emp_moments(z,deg)
    [mean(z.^i) for i in 0:deg]
end

@testset "admissibility" begin
    s = FSE.uniform_moments(10,0,1)
    @assert s == [1/(i+1) for i in 0:10]
    zz = rand(25)
    #Test admissibility of moment sequences
    for n in 1:20
        for a in range(-1,-.2,5)
            for b in range(.2,1,5)
                st = FSE.uniform_moments(n,a,b)
                @assert FSE.is_admissible(st,a,b)
                sz = emp_moments((b-a)*zz .+ a,n)
                @assert FSE.is_admissible(sz,a,b)
            end
        end
    end

end
@testset "moment_bounds" begin
    #Test conditional moment bounds
    for a in range(-1,-.2,5)
        for b in range(.2,1,5)
            l,u=FSE.conditional_moment_bounds([1.],a,b)
            @assert l == a && u == b
            m = (a+b)/2
            l,u=FSE.conditional_moment_bounds([1.,m],a,b)
            @assert l >= m^2
        end
    end

    for n in 1:10
        s = FSE.legendre_moments(n)
        l,u=FSE.conditional_moment_bounds(s)
        #Different way of computing the same quantity
        lwr = FSE.moments_qr(FSE.lower_representation(s)...,n+1)[end]
        upr= FSE.moments_qr(FSE.upper_representation(s)...,n+1)[end]
        @assert (l ≈ lwr) && (u ≈ upr)
    end

end

@testset "markov_bounds" begin
    for a in range(-1,-.2,5)
        for b in range(.2,1,5)
            for s1 in range(a+.01,b-.01,3)
                for t in range(a+.01,b-.01,3)
                    l,u=FSE.markov_bound([1.0,s1],t,a,b)
                    @assert l ≈ max((a-s1)/(t-a) + 1,0)
                    @assert u ≈ min((s1-b)/(t-b),1)
                end
            end
        end
    end
    a=-.3
    b=.9
    for n in 1:6
        for t in range(a+.01,b-.01,10)

            s = FSE.uniform_moments(n,a,b)
            l,u=FSE.markov_bound_dual(s,t,a=a,b=b,ng=5000)
            lw,up=FSE.markov_bound(s,t,a,b)
            @assert (abs(l -lw) < 1e-3 ) && (abs(u - up) < 1e-3)
        end
    end
end
