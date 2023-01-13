using Statistics

function accuracy(y_true, y_pred)
    mean(y_true .== y_pred)
end

function mse(y_true, y_pred)
    sum((y_true .- y_pred) .^ 2) / length(y_true)
end