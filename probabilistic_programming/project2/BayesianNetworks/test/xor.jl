using Test, BayesianNetworks

@testset "XOR - gradient training" begin

    network = BayesianNetworks.Xor.make_network()
    X, y = BayesianNetworks.Xor.make_dataset(5000)
    scaler = BayesianNetworks.std_scaler_fit(X)
    X = BayesianNetworks.std_scaler_transform(scaler, X)
    y_onehot = BayesianNetworks.Xor.encode_labels(y)
    network, _ = BayesianNetworks.train_model_using_gd(network, X, y_onehot, batchsize=512, epochs=100)
    y_pred = BayesianNetworks.Xor.predict(network, X)
    @test BayesianNetworks.accuracy(y, y_pred) > 0.95

    test_X, test_y = BayesianNetworks.Xor.make_dataset(5000)
    test_X = BayesianNetworks.std_scaler_transform(scaler, test_X)
    test_y_pred = BayesianNetworks.Xor.predict(network, test_X)
    @test BayesianNetworks.accuracy(test_y, test_y_pred) > 0.95
end