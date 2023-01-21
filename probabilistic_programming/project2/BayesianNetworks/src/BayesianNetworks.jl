module BayesianNetworks

using Gen, LinearAlgebra, ProgressMeter, Flux

include("preprocessing.jl")
include("training.jl")
include("metrics.jl")
include("xor.jl")

end
