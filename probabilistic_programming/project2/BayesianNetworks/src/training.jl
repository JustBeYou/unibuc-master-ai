using Flux, ProgressMeter, StatsBase

function train_model_using_gd(
    model, X, y;
    val_X=nothing, val_y=nothing,
    optimizer=Flux.Adam(0.01),
    criterion=Flux.crossentropy,
    epochs=10,
    batchsize=8,
    device=cpu
)
    model = model |> device
    dataloader = Flux.DataLoader((X, y) |> device, batchsize=batchsize, shuffle=true)
    optimizer = Flux.setup(optimizer, model)

    losses, val_losses = [], []
    @showprogress for _ in 1:epochs
        for (X_batch, y_batch) in dataloader
            loss, (gradients,) = Flux.withgradient(model) do m
                y_batch_hat = m(X_batch)
                criterion(y_batch_hat, y_batch)
            end

            if !isnothing(val_X)
                val_y_hat = model(val_X)
                val_loss = criterion(val_y_hat, val_y)
                push!(val_losses, val_loss)
            end

            Flux.update!(optimizer, model, gradients)
            push!(losses, loss)
        end
    end

    return model, losses, val_losses
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

function make_constraints(y)
    constraints = Gen.choicemap()
    for i in eachindex(y)
        constraints[:y=>i] = y[i]
    end
    return constraints
end

function constrain_parameters(constraints, selector_indices, trace)
    old_choices = get_choices(trace)
    for i in selector_indices
        constraints[:parameters=>i=>:value] = old_choices[:parameters=>i=>:value]
    end
    return constraints
end

function sample_batch(X, y, batchsize)
    X_length = last(size(X))
    batch_idx = StatsBase.sample(1:X_length, batchsize; replace=false)
    batch_X, batch_y = X[:, batch_idx], y[batch_idx]
    return batch_X, batch_y
end

function infer_models_using_mcmc(
    model, X, y, labels;
    val_X=nothing, val_y=nothing,
    miu=0.0, sigma=1.0,
    batchsize=256,
    burnin_epochs=100, inference_epochs=10,
    ascending_select=true,
    criterion=Flux.crossentropy
)
    model = fmap(f64, model)
    parameters_0, reconstruct = Flux.destructure(model)
    N = length(parameters_0)

    batch_X, batch_y = sample_batch(X, y, batchsize)
    constraints = make_constraints(batch_y)

    trace, _ = Gen.generate(network_parameters_apriori, (batch_X, N, reconstruct, miu, sigma), constraints)
    selector_indices = ascending_select ? (1:N) : (N:-1:1)
    selectors = [Gen.select(:parameters => i => :value) for i in selector_indices]

    if !isnothing(val_X)
        val_y_hot = BayesianNetworks.encode_labels(val_y, labels)
    end

    losses, val_losses, accuracies = [], [], []
    @showprogress for _ in 1:burnin_epochs
        batch_X, batch_y = sample_batch(X, y, batchsize)
        constraints = make_constraints(batch_y)
        constraints = constrain_parameters(constraints, selector_indices, trace)
        trace, _ = Gen.generate(network_parameters_apriori, (batch_X, N, reconstruct, miu, sigma), constraints)

        for selector in selectors
            trace, _ = Gen.mh(trace, selector)
        end

        batch_y_hot = BayesianNetworks.encode_labels(batch_y, labels)
        model = model_from_choices(N, reconstruct, trace)
        loss = criterion(model(batch_X), batch_y_hot)
        push!(losses, loss)

        if !isnothing(val_X)
            val_y_hat = model(val_X)
            val_loss = criterion(val_y_hat, val_y_hot)
            push!(val_losses, val_loss)
        end

        y_pred = BayesianNetworks.predict(model, batch_X, labels)
        accuracy = BayesianNetworks.accuracy(batch_y, y_pred)
        push!(accuracies, accuracy)
    end

    models = []
    @showprogress for _ in 1:inference_epochs
        batch_X, batch_y = sample_batch(X, y, batchsize)
        constraints = make_constraints(batch_y)
        constraints = constrain_parameters(constraints, selector_indices, trace)
        trace, _ = Gen.generate(network_parameters_apriori, (batch_X, N, reconstruct, miu, sigma), constraints)

        for selector in selectors
            trace, _ = Gen.mh(trace, selector)
        end

        model = model_from_choices(N, reconstruct, trace)
        push!(models, model)
    end

    return models, losses, val_losses, accuracies
end