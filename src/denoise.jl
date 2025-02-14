#Find the closest admissible moment sequence on [a,b]
#Vaguely inspired by Lasserre (2009) but implemented differently
#See also Wu & Yang, Optimal Estimation Of Gaussian Mixtures Via Denoised
#Method Of Moments
function denoise(s;a=-1.,b=1.,force_s0=true,w=ones(length(s)))
    n = length(s)
    if isodd(n)
        p=div(n-1,2) #max degree
        sX = p+1
        sY = p
    else
        p=div(n-2,2)
        sX = p+1
        sY = p+1
    end
    n=length(s)
    model = Model(() -> Hypatia.Optimizer(verbose = false))
    @variable(model,X[1:sX,1:sX],PSD)
    @variable(model,Y[1:sY,1:sY],PSD)
    @variable(model,z[1:length(s)])
    #D = Diagonal(w)
    
    D = Diagonal(normalize(sqrt.(w),1))
    @objective(model,Min,sum(abs2.(D*(z-s))))
    if force_s0
        @constraint(model, z[1] == s[1])
    end
#    @constraint(model,X[1,1] == s[1])
    if isodd(n)
        #Moment matrix 
        for i in 0:(p)
            for j in 0:(p)
                d = i +j + 1
                @constraint(model,X[i+1,j+1] == z[d])
            end
        end
        #Localised moment matrix
        for i in 0:(p-1)
            for j in 0:(p-1)
                d=i+j+1
                @constraint(model,Y[i+1,j+1] == -a*b*z[d] + (a+b)*z[d+1]-z[d+2])
            end
        end
    else
        for i in 0:(p)
            for j in 0:(p)
                d = i +j + 1
                @constraint(model,X[i+1,j+1] == z[d+1]-a*z[d])
                @constraint(model,Y[i+1,j+1] == b*z[d]-z[d+1])
            end
        end
    end
    optimize!(model)
    value.(z)
end
