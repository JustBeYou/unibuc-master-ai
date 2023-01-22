module SanityCheck

using Flux

function make_dataset(n)
    X = rand(Float32, 2, n)
    y = tanh.(X[1, :] + X[2, :])
    y = 1 ./ (1 .+ exp.(-(y .+ y)))
    y = map(yi -> yi > 0.5 ? 1 : 2, y)
    return X, y
end

const labels = [1, 2]

function make_network()
    return Chain(
        Dense(2 => 2, tanh),
        Dense(2 => 2),
        softmax,
    )
end

end