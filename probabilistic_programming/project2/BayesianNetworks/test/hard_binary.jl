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