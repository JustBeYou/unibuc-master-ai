module BayesianNetworks

using Gen, LinearAlgebra, ProgressMeter, Flux

include("preprocessing.jl")
include("training.jl")
include("prediction.jl")
include("metrics.jl")
include("examples/xor.jl")
include("examples/iris.jl")
include("examples/sanity_check.jl")
include("examples/hard_binary.jl")
include("examples/hard_multi.jl")
@load_generated_functions()

end
