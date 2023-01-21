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

    losses, accuracies = [], []
    @showprogress for _ in 1:epochs
        for (X_batch, y_batch) in dataloader
            accuracy = 0
            loss, (gradients,) = Flux.withgradient(model) do m
                y_batch_hat = m(X_batch)
                accuracy = BayesianNetworks.accuracy(y_batch, y_batch_hat)
                criterion(y_batch_hat, y_batch)
            end

            Flux.update!(optimizer, model, gradients)
            push!(losses, loss)
            push!(accuracies, accuracy)
        end
    end

    return model, losses, accuracies
end

@gen function single_parameter_apriori(miu, sigma)
    value ~ Gen.normal(miu, sigma)
end
map_parameters_apriori = Gen.Map(single_parameter_apriori)

@gen function predictions_apriori(pred)
    for i in 1:last(size(pred))
        Gen.@trace(Gen.categorical(pred[:, i]), i)
    end
end

@gen (static) function network_parameters_apriori(X, N, reconstruct, miu, sigma)
    parameters ~ map_parameters_apriori(fill(miu, N), fill(sigma, N))

    network = reconstruct(parameters)
    pred = network(X)
    y ~ predictions_apriori(pred)
end

function model_from_choices(N, reconstruct, trace)
    choices = Gen.get_choices(trace)
    parameters = [choices[:parameters=>i=>:value] for i in 1:N]
    return reconstruct(parameters)
end

function infer_models_using_mcmc(
    model, X, y, labels;
    miu=0.0, sigma=1.0,
    burnin_epochs=50, inference_epochs=10,
    ascending_select=true,
    criterion=Flux.crossentropy
)
    model = fmap(f64, model)
    parameters_0, reconstruct = Flux.destructure(model)
    N = length(parameters_0)

    constraints = Gen.choicemap()
    for i in eachindex(y)
        constraints[:y=>i] = y[i]
    end

    trace, _ = Gen.generate(network_parameters_apriori, (X, N, reconstruct, miu, sigma), constraints)
    selector_indices = ascending_select ? (1:N) : (N:-1:1)
    selectors = [Gen.select(:parameters => i => :value) for i in selector_indices]

    losses, accuracies = [], []
    @showprogress for _ in 1:burnin_epochs
        for selector in selectors
            trace, _ = Gen.mh(trace, selector; check=true, observations=constraints)
        end

        model = model_from_choices(N, reconstruct, trace)
        y_pred = BayesianNetworks.predict(model, X, labels)

        accuracy = BayesianNetworks.accuracy(y, y_pred)
        push!(accuracies, accuracy)

        loss = criterion(y_pred, y)
        push!(losses, loss)
    end

    models = []
    @showprogress for _ in 1:inference_epochs
        for selector in selectors
            trace, _ = Gen.mh(trace, selector; check=true, observations=constraints)
        end

        model = model_from_choices(N, reconstruct, trace)
        push!(models, model)
    end

    return models, losses, accuracies
end