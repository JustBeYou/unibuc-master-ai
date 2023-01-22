module Iris

using MLJBase, Flux

function make_dataset()
    X, y = MLJBase.@load_iris()
    # We need a matrix as input
    X = reduce(hcat, collect(X))'
    y = map(yi -> labels_mapping[yi], y)

    N = last(size(X))
    idx, val_idx, test_idx = MLJBase.partition(1:N, 0.8, 0.1, shuffle=true, stratify=y)

    train_X, train_y = X[:, idx], y[idx]
    val_X, val_y = X[:, val_idx], y[val_idx]
    test_X, test_y = X[:, test_idx], y[test_idx]

    return train_X, val_X, test_X, train_y, val_y, test_y
end

const labels_mapping = Dict(
    "setosa" => 1,
    "versicolor" => 2,
    "virginica" => 3,
)
const labels = [1, 2, 3]

function make_network()
    return Chain(
        Dense(4 => 5, relu),
        BatchNorm(5),
        Dense(5 => 3),
        softmax,
    )
end

end