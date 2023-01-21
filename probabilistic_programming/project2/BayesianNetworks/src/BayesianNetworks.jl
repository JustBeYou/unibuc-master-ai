module BayesianNetworks

using Gen, LinearAlgebra, ProgressMeter, Flux

include("preprocessing.jl")
include("training.jl")
include("metrics.jl")
include("xor.jl")

@gen function bayes_nn(X, N, reconstruct, miu, sigma)
    parameters = Vector{Float64}(undef, N)
    for i in 1:N
        parameters[i] = Gen.@trace(Gen.normal(miu, sigma), :parameters => i)
    end

    network = reconstruct(parameters)
    pred = network(X)

    y = Vector{Bool}(undef, last(size(X)))
    for i in 1:last(size(y))
        y[i] = Gen.@trace(Gen.bernoulli(pred[1, i]), :y => i)
    end
end

function check_gen_training()
    dataset_size = 100
    X, y = BayesianNetworks.Xor.make_dataset(dataset_size)

    network = BayesianNetworks.Xor.make_network()
    network = fmap(f64, network)

    parameters_0, reconstruct = Flux.destructure(network)
    N = length(parameters_0)

    constraints = Gen.choicemap()
    for i in eachindex(y)
        constraints[:y=>i] = y[i]
    end

    trace, _ = Gen.generate(bayes_nn, (X, N, reconstruct, 0.0, 3.0), constraints)
    selectors = [Gen.select(:parameters => i) for i in 1:N]

    burn_in = 5000
    inference_samples = 1000
    @showprogress for _ in 1:burn_in
        for selector in selectors
            trace, _ = Gen.mh(trace, selector; check=true, observations=constraints)
        end
    end

    mean_y_pred = zeros(last(size(X)))
    @showprogress for _ in 1:inference_samples
        for selector in selectors
            trace, _ = Gen.mh(trace, selector; check=true, observations=constraints)
        end

        choices = Gen.get_choices(trace)
        parameters = [choices[:parameters=>i] for i in 1:N]
        network = reconstruct(parameters)
        y_pred = BayesianNetworks.Xor.predict(network, X)

        mean_y_pred += y_pred
    end

    mean_y_pred ./= inference_samples
    mean_y_pred = round.(mean_y_pred)


    acc = BayesianNetworks.accuracy(y, mean_y_pred)
    println("Accuracy: $acc")
end

end
