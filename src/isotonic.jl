#Basic isotonic regression for cleaning up the cdfs
function isotonic(x,y)
    model = Model(() -> Hypatia.Optimizer(verbose = false))
    keep = findall(isfinite.(y))
    n = length(x[keep])
    @variable(model, f[1:n])
    @objective(model, Min, sum((y[keep]-f).^2))
    @constraint(model,f[2:end]-f[1:(end-1)] >= 0)
    @constraint(model,f .>= 0)
    @constraint(model,f .<= 1)
    optimize!(model)
    (xs=x[keep],ys=value.(f))
end


