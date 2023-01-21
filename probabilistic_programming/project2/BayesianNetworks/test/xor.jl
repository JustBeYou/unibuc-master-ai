using Test, BayesianNetworks, StatsBase

@testset "XOR - gradient training" begin
    network = BayesianNetworks.Xor.make_network()
    X, y = BayesianNetworks.Xor.make_dataset(5000)
    scaler = BayesianNetworks.std_scaler_fit(X)
    X = BayesianNetworks.std_scaler_transform(scaler, X)
    y_onehot = BayesianNetworks.encode_labels(y, BayesianNetworks.Xor.labels)
    network, _ = BayesianNetworks.train_model_using_gd(network, X, y_onehot, batchsize=512, epochs=100)
    y_pred = BayesianNetworks.predict(network, X, BayesianNetworks.Xor.labels)
    @test BayesianNetworks.accuracy(y, y_pred) > 0.95

    test_X, test_y = BayesianNetworks.Xor.make_dataset(5000)
    test_X = BayesianNetworks.std_scaler_transform(scaler, test_X)
    test_y_pred = BayesianNetworks.predict(network, test_X, BayesianNetworks.Xor.labels)
    @test BayesianNetworks.accuracy(test_y, test_y_pred) > 0.95
end

@testset "XOR - MCMC inference" begin
    network = BayesianNetworks.Xor.make_network()
    X, y = BayesianNetworks.Xor.make_dataset(1000)
    scaler = BayesianNetworks.std_scaler_fit(X)
    X = BayesianNetworks.std_scaler_transform(scaler, X)
    networks, _, _ = BayesianNetworks.infer_models_using_mcmc(network, X, y, BayesianNetworks.Xor.labels)
    mean_y_pred = [BayesianNetworks.predict(network, X, BayesianNetworks.Xor.labels) for network in networks]
    mean_y_pred = round.(mean(mean_y_pred))
    @test BayesianNetworks.accuracy(y, mean_y_pred) > 0.85

    test_X, test_y = BayesianNetworks.Xor.make_dataset(1000)
    test_X = BayesianNetworks.std_scaler_transform(scaler, test_X)
    mean_test_y_pred = [BayesianNetworks.predict(network, test_X, BayesianNetworks.Xor.labels) for network in networks]
    mean_test_y_pred = round.(mean(mean_test_y_pred))
    @test BayesianNetworks.accuracy(test_y, mean_test_y_pred) > 0.85
end