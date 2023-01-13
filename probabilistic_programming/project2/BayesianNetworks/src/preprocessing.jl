using StatsBase

function std_scaler_fit(X)
    StatsBase.fit(ZScoreTransform, X, dims=2)
end

function std_scaler_transform(scaler, X)
    StatsBase.transform(scaler, X)
end