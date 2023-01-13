module Linear

using Flux

function make_dataset(n, a, b)
    X = rand(-100:100, 1, n)
    y = [a * column[1] + b >= 0 for column in eachcol(X)]
    return X, y
end

const labels = [true, false]

function encode_labels(y)
    return Flux.onehotbatch(y, labels)
end

function make_network()
    return Chain(
        Dense(1 => 2),
        softmax
    )
end

function predict(model, X, device=cpu)
    y_pred = model(X |> device) |> cpu
    return y_pred[1, :] .> 0.5
end

end