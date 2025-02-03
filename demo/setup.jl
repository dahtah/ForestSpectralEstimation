#Run once to set up environment
#Have to do this because KirchhoffForests is unregistered
import Pkg
Pkg.activate(".")
Pkg.add("https://github.com/dahtah/KirchhoffForests.jl")
Pkg.add("..") #Add current package
