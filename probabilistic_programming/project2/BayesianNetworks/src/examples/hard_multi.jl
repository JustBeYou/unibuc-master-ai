module HardMultiClassification

using MLJBase, Flux

function make_dataset(n)
    X, y = MLJBase.make_blobs(n, 10; centers=10, cluster_std=[rand() * 5 for _ in 1:10], as_table=false)
    # Flux requires the batch to be the last dimension
    X = permutedims(X, [2, 1])

    N = last(size(X))
    idx, val_idx, test_idx = MLJBase.partition(1:N, 0.8, 0.1, shuffle=true, stratify=y)

    train_X, train_y = X[:, idx], y[idx]
    val_X, val_y = X[:, val_idx], y[val_idx]
    test_X, test_y = X[:, test_idx], y[test_idx]

    return train_X, val_X, test_X, train_y, val_y, test_y
end

const labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

function make_network()
    return Chain(
        Dense(10 => 5, relu),
        BatchNorm(5),
        Dense(5 => 10),
        softmax,
    )
end

end