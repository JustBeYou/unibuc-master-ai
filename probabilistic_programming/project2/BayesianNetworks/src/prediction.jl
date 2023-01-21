using Flux

function encode_labels(y, labels)
    return Flux.onehotbatch(y, labels)
end

function predict(model, X, labels; device=cpu)
    y_pred = model(X |> device) |> cpu
    return Flux.onecold(y_pred, labels)
end