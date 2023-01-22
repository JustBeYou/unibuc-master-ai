using Test, BayesianNetworks, StatsBase

@testset "Hard binary classification: two moons - gradient training" begin
    network = BayesianNetworks.HardBinaryClassification.make_network()
    X, y = BayesianNetworks.HardBinaryClassification.make_dataset(5000)
    scaler = BayesianNetworks.std_scaler_fit(X)
    X = BayesianNetworks.std_scaler_transform(scaler, X)
    y_onehot = BayesianNetworks.encode_labels(y, BayesianNetworks.HardBinaryClassification.labels)
    network, _, _ = BayesianNetworks.train_model_using_gd(network, X, y_onehot, batchsize=512, epochs=100)
    y_pred = BayesianNetworks.predict(network, X, BayesianNetworks.HardBinaryClassification.labels)
    @test BayesianNetworks.accuracy(y, y_pred) > 0.93

    test_X, test_y = BayesianNetworks.HardBinaryClassification.make_dataset(5000)
    test_X = BayesianNetworks.std_scaler_transform(scaler, test_X)
    test_y_pred = BayesianNetworks.predict(network, test_X, BayesianNetworks.HardBinaryClassification.labels)
    @test BayesianNetworks.accuracy(test_y, test_y_pred) > 0.93
end

@testset "Hard binary classification: two moons - MCMC inference" begin
    network = BayesianNetworks.HardBinaryClassification.make_network()
    X, y = BayesianNetworks.HardBinaryClassification.make_dataset(1000)
    scaler = BayesianNetworks.std_scaler_fit(X)
    X = BayesianNetworks.std_scaler_transform(scaler, X)
    networks, _, _, _ = BayesianNetworks.infer_models_using_mcmc(network, X, y, BayesianNetworks.HardBinaryClassification.labels)
    mean_y_pred = [BayesianNetworks.predict(network, X, BayesianNetworks.HardBinaryClassification.labels) for network in networks]
    mean_y_pred = round.(mean(mean_y_pred))
    @test BayesianNetworks.accuracy(y, mean_y_pred) > 0.85

    test_X, test_y = BayesianNetworks.HardBinaryClassification.make_dataset(1000)
    test_X = BayesianNetworks.std_scaler_transform(scaler, test_X)
    mean_test_y_pred = [BayesianNetworks.predict(network, test_X, BayesianNetworks.HardBinaryClassification.labels) for network in networks]
    mean_test_y_pred = round.(mean(mean_test_y_pred))
    @test BayesianNetworks.accuracy(test_y, mean_test_y_pred) > 0.85
end