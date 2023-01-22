using Test, BayesianNetworks, StatsBase

@testset "Hard binary classification: blobs - gradient training" begin
    X, _, test_X, y, _, test_y = BayesianNetworks.HardMultiClassification.make_dataset(1000)

    network = BayesianNetworks.HardMultiClassification.make_network()
    scaler = BayesianNetworks.std_scaler_fit(X)
    X = BayesianNetworks.std_scaler_transform(scaler, X)
    y_onehot = BayesianNetworks.encode_labels(y, BayesianNetworks.HardMultiClassification.labels)
    network, _, _ = BayesianNetworks.train_model_using_gd(network, X, y_onehot, batchsize=10, epochs=50)
    y_pred = BayesianNetworks.predict(network, X, BayesianNetworks.HardMultiClassification.labels)
    @test BayesianNetworks.accuracy(y, y_pred) > 0.90

    test_X = BayesianNetworks.std_scaler_transform(scaler, test_X)
    test_y_pred = BayesianNetworks.predict(network, test_X, BayesianNetworks.HardMultiClassification.labels)
    @test BayesianNetworks.accuracy(test_y, test_y_pred) > 0.90
end

@testset "Hard binary classification: blobs - MCMC inference" begin
    X, _, test_X, y, _, test_y = BayesianNetworks.HardMultiClassification.make_dataset(1000)

    network = BayesianNetworks.HardMultiClassification.make_network()
    scaler = BayesianNetworks.std_scaler_fit(X)
    X = BayesianNetworks.std_scaler_transform(scaler, X)
    networks, _, _, _ = BayesianNetworks.infer_models_using_mcmc(network, X, y, BayesianNetworks.HardMultiClassification.labels)
    mean_y_pred = [BayesianNetworks.predict(network, X, BayesianNetworks.HardMultiClassification.labels) for network in networks]
    mean_y_pred = round.(mean(mean_y_pred))
    @test BayesianNetworks.accuracy(y, mean_y_pred) > 0.84

    test_X = BayesianNetworks.std_scaler_transform(scaler, test_X)
    mean_test_y_pred = [BayesianNetworks.predict(network, test_X, BayesianNetworks.HardMultiClassification.labels) for network in networks]
    mean_test_y_pred = round.(mean(mean_test_y_pred))
    @test BayesianNetworks.accuracy(test_y, mean_test_y_pred) > 0.84
end