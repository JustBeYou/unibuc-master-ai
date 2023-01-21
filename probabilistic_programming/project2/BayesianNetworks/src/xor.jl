module Xor

using Flux

function make_dataset(n)
    X = rand(Float32, 2, n)
    y = [xor(column[1] > 0.5, column[2] > 0.5) ? 1 : 2 for column in eachcol(X)]
    return X, y
end

const labels = [1, 2]

function make_network()
    return Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2),
        softmax,
    )
end

end