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