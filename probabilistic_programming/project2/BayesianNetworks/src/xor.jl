module Xor

using Flux

function make_dataset(n)
    X = rand(Float32, 2, n)
    y = [xor(column[1] > 0.5, column[2] > 0.5) for column in eachcol(X)]
    return X, y
end

const labels = [true, false]

function encode_labels(y)
    return Flux.onehotbatch(y, labels)
end

function make_network()
    return fmap(f64, Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2),
        softmax,
    ))
end

function predict(model, X, device=cpu)
    y_pred = model(X |> device) |> cpu
    return y_pred[1, :] .> 0.5
end

end