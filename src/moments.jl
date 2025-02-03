#Some code for handling (classical) moments
#In this file a moment sequence is represented by a vector
#s = [s[0],s[1],...,s[k-1]]
#where s[i] = E(x^i)
#Note that generally s[0] = 1

#Classical moment matrix
function moment_matrix(s)
    m = length(s)-1
    n = iseven(m) ? div(m,2) : div(m-1,2)
    s = OffsetArrays.Origin(0)(s)
    [s[i + j] for i in 0:n,j in 0:n]
end


#Given a moment sequence, find upper and lower bounds for the next moment
#Uses Hankel determinants; see e.g. Dette & Studden, Theory of Canonical Moments
#p. 20
function conditional_moment_bounds(s,a=-1.0,b=1.0)
    m = length(s)-1
    s = OffsetArrays.Origin(0)(s)
    if m == 1
        up=a+b-a*b
        lw=s[1]^2
        return lw,up
    end
    if isodd(m)
        n = div(m-1,2)
        Hl=[s[i + j] for i in 0:n,j in 0:n]
        z=[s[1+n+i] for i in 0:n]


        lw=z'*(Symmetric(Hl)\z)
        Hu=[(a+b)*s[i+j+1] - s[i+j+2] - a*b*s[i+j] for i in 0:(n-1),j in 0:(n-1)]
        z=[(a+b)*s[i+n+1] - s[i+n+2] - a*b*s[i+n] for i in 0:(n-1)]
        up= - z'*(Hu\z) + (a+b)*s[2n+1]-a*b*s[2n]
        lw,up
    else
        n = div(m-2,2)
        Hl=[s[i + j + 1] - a* s[i+j] for i in 0:n,j in 0:n]
        z=[s[n+1 + j + 1] - a* s[n+1+j] for j in 0:n]
        lw=z'*(Hl\z)+a*s[end]
        Hu=[b*s[i + j] - s[i+j+1] for i in 0:n,j in 0:n]
        z=[b*s[n+1 + j] - s[n+1+j+1] for j in 0:n]
        up=-z'*(Hu\z)+b*s[end]
        lw,up
    end
end

#Various matrices used to ensure that the moment sequence s is admissible.
#see e.g. Dette & Studden, Theory of Canonical Moments p. 20
function Hmat(s,a=-1,b=1)
    m = length(s)-1
    s = OffsetArrays.Origin(0)(s)
    if iseven(m)
        n = div(m,2)
        Hl=[s[i + j] for i in 0:n,j in 0:n]
        Hu=[(a+b)*s[i+j+1] - s[i+j+2] - a*b*s[i+j] for i in 0:(n-1),j in 0:(n-1)]
        Hl,Hu
    else
        n = div(m-1,2)
        Hl=[s[i + j + 1] - a* s[i+j] for i in 0:n,j in 0:n]
        Hu=[b*s[i + j] - s[i+j+1] for i in 0:n,j in 0:n]
        Hl,Hu
    end
end

#Verify that moment sequence s is admissible on the interval [a,b]
#ie that there exists a measure with support on [a,b] that has s has its (truncated) moment vector
#see Schmüdgen, The Moment Problem, p. 230
function is_admissible(s,a=-1.0,b=1.0)
    all(map(isposdef,Hmat(s,a,b)))
end

#find largest degree k such that s[1:k] is admissible
#stupid implementation, use only for low degrees!
function admissible_subset(s,a=-1.0,b=1.0)
    k = length(s)
    while k > 0
        if is_admissible(s[1:k])
            break
        else
            k -= 1
        end
    end
    k,s[1:k]
end
