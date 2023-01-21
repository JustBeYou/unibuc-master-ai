using Flux, ProgressMeter

function train_model_using_gd(
    model, X, y;
    optimizer=Flux.Adam(0.01),
    criterion=Flux.crossentropy,
    epochs=10,
    batchsize=8,
    device=cpu
)
    model = model |> device
    dataloader = Flux.DataLoader((X, y) |> device, batchsize=batchsize, shuffle=true)
    optimizer = Flux.setup(optimizer, model)

    losses = []
    @showprogress for _ in 1:epochs
        for (X_batch, y_batch) in dataloader
            loss, (gradients,) = Flux.withgradient(model) do m
                y_batch_hat = m(X_batch)
                criterion(y_batch_hat, y_batch)
            end

            Flux.update!(optimizer, model, gradients)
            push!(losses, loss)
        end
    end

    return model, losses
end

@gen function network_parameters_posterior(X, N, reconstruct, miu, sigma)
    parameters = Vector{Float64}(undef, N)
    for i in 1:N
        parameters[i] = Gen.@trace(Gen.normal(miu, sigma), :parameters => i)
    end

    network = reconstruct(parameters)
    pred = network(X)

    y = Vector{Int}(undef, last(size(X)))
    for i in 1:last(size(y))
        y[i] = Gen.@trace(Gen.categorical(pred[:, i]), :y => i)
    end
end

function infer_models_using_mcmc(
    model, X, y;
    miu=0.0, sigma=1.0,
    burnin_epochs=1000, inference_epochs=100,
    ascending_select=true
)
    model = fmap(f64, model)
    parameters_0, reconstruct = Flux.destructure(model)
    N = length(parameters_0)

    constraints = Gen.choicemap()
    for i in eachindex(y)
        constraints[:y=>i] = y[i]
    end

    trace, _ = Gen.generate(network_parameters_posterior, (X, N, reconstruct, miu, sigma), constraints)
    selector_indices = ascending_select ? (1:N) : (N:-1:1)
    selectors = [Gen.select(:parameters => i) for i in selector_indices]

    @showprogress for _ in 1:burnin_epochs
        for selector in selectors
            trace, _ = Gen.mh(trace, selector; check=true, observations=constraints)
        end
    end

    models = []
    @showprogress for _ in 1:inference_epochs
        for selector in selectors
            trace, _ = Gen.mh(trace, selector; check=true, observations=constraints)
        end

        choices = Gen.get_choices(trace)
        parameters = [choices[:parameters=>i] for i in 1:N]
        model = reconstruct(parameters)
        push!(models, model)
    end

    return models, []
end