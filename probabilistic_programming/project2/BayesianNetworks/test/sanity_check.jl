using Test, BayesianNetworks, StatsBase

@testset "Sanity Check - gradient training" begin
    network = BayesianNetworks.SanityCheck.make_network()
    X, y = BayesianNetworks.SanityCheck.make_dataset(5000)
    y_onehot = BayesianNetworks.encode_labels(y, BayesianNetworks.SanityCheck.labels)
    network, _, _ = BayesianNetworks.train_model_using_gd(network, X, y_onehot, batchsize=512, epochs=100)
    y_pred = BayesianNetworks.predict(network, X, BayesianNetworks.SanityCheck.labels)
    @test BayesianNetworks.accuracy(y, y_pred) > 0.95

    test_X, test_y = BayesianNetworks.SanityCheck.make_dataset(5000)
    test_y_pred = BayesianNetworks.predict(network, test_X, BayesianNetworks.SanityCheck.labels)
    @test BayesianNetworks.accuracy(test_y, test_y_pred) > 0.95
end

@testset "Sanity Check - MCMC inference" begin
    network = BayesianNetworks.SanityCheck.make_network()
    X, y = BayesianNetworks.SanityCheck.make_dataset(1000)
    networks, _, _, _ = BayesianNetworks.infer_models_using_mcmc(network, X, y, BayesianNetworks.SanityCheck.labels)
    mean_y_pred = [BayesianNetworks.predict(network, X, BayesianNetworks.SanityCheck.labels) for network in networks]
    mean_y_pred = round.(mean(mean_y_pred))
    @test BayesianNetworks.accuracy(y, mean_y_pred) > 0.85

    test_X, test_y = BayesianNetworks.SanityCheck.make_dataset(1000)
    mean_test_y_pred = [BayesianNetworks.predict(network, test_X, BayesianNetworks.SanityCheck.labels) for network in networks]
    mean_test_y_pred = round.(mean(mean_test_y_pred))
    @test BayesianNetworks.accuracy(test_y, mean_test_y_pred) > 0.85
end