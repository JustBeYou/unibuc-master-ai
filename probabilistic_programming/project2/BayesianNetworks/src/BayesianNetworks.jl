module BayesianNetworks

using Gen, LinearAlgebra, ProgressMeter, Flux

include("preprocessing.jl")
include("training.jl")
include("metrics.jl")
include("xor.jl")
include("linear.jl")

@gen function bayes_nn(X, N, reconstruct, miu, sigma)
    parameters = Vector{Float64}(undef, N)
    for i in 1:N
        parameters[i] = @trace(normal(miu, sigma), :parameters => i)
    end

    network = reconstruct(parameters)
    pred = network(X)

    y = Vector{Bool}(undef, last(size(X)))
    for i in eachindex(y)
        y[i] = @trace(bernoulli(pred[i]), :y => i)
    end
end

function check_gen_training()
    a = 10
    b = 5
    dataset_size = 10
    X, y = BayesianNetworks.Linear.make_dataset(dataset_size, a, b)

    network = BayesianNetworks.Linear.make_network()
    network = fmap(f64, network)

    parameters_0, reconstruct = Flux.destructure(network)
    N = length(parameters_0)

    constraints = choicemap()
    for i in eachindex(y)
        constraints[:y=>i] = y[i]
    end

    trace, _ = generate(bayes_nn, (X, N, reconstruct, 0.0, 1.0), constraints)
    selectors = [Gen.select(:parameters=>i) for i in 1:N]

    # burn in
    burn_in = 0
    @showprogress for _ in 1:burn_in
        for selector in selectors
            trace, _ = mh(trace, selector, check=true, observations=constraints)
        end 
    end

    # sample after convergence
    samples = 5
    thin = 1
    y_probs = zeros(2, dataset_size)::Matrix{Float64}
    acc_history = []

    @showprogress for _ in 1:samples
        for _ in 1:thin
            trace, _ = mh(trace, Gen.select(), check=true, observations=constraints)
        end
        choices = get_choices(trace)
        network = reconstruct([choices[:parameters => i] for i in 1:N])

        tmp_y_probs = network(X)
        println([choices[:parameters => i] for i in 1:N])
        # println(tmp_y_probs)
        tmp_y_pred = tmp_y_probs[1, :] .> 0.5
        push!(acc_history, accuracy(y, tmp_y_pred))
        tmp_y_pred = tmp_y_probs[2, :] .> 0.5
        push!(acc_history, accuracy(y, tmp_y_pred))

        y_probs .+= tmp_y_probs
    end

    println("Accuracy history $acc_history")

    y_probs ./= samples
    y_pred = y_probs[1, :] .> 0.5

    acc = accuracy(y, y_pred)
    println("Accuracy: $acc")
end

end
