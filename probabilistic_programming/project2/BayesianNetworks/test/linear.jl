using Test, BayesianNetworks

@testset "Linear classification - gradient training" begin

    a = 0.5
    b = 0.2
    X, y = BayesianNetworks.Linear.make_dataset(5000, a, b)

    network = BayesianNetworks.Linear.make_network()
    y_onehot = BayesianNetworks.Xor.encode_labels(y)
    network, _ = BayesianNetworks.train_model(network, X, y_onehot, batchsize=512, epochs=100)
    y_pred = BayesianNetworks.Linear.predict(network, X)
    @test BayesianNetworks.accuracy(y, y_pred) > 0.95

    test_X, test_y = BayesianNetworks.Linear.make_dataset(5000, a, b)
    test_y_pred = BayesianNetworks.Linear.predict(network, test_X)
    @test BayesianNetworks.accuracy(test_y, test_y_pred) > 0.95
end