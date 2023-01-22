module BayesianNetworks

using Gen, LinearAlgebra, ProgressMeter, Flux

include("preprocessing.jl")
include("training.jl")
include("prediction.jl")
include("metrics.jl")
include("xor.jl")
include("sanity_check.jl")
@load_generated_functions()

end
