module HardBinaryClassification

using MLJBase, Flux

function make_dataset(n)
    X, y = MLJBase.make_moons(n; noise=0.3, as_table=false)
    # Flux requires the batch to be the last dimension
    X = permutedims(X, [2, 1])
    # We count classes from 1
    y .+= 1
    return X, y
end

const labels = [1, 2]

function make_network()
    return Chain(
        Dense(2 => 5, relu),
        BatchNorm(5),
        Dense(5 => 5, relu),
        BatchNorm(5),
        Dense(5 => 2),
        softmax,
    )
end

end