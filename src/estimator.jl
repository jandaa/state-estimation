using CoordinateTransformations, Rotations, StaticArrays
using LinearAlgebra
using TransformUtils

include("lie_groups.jl")

# Define batch estimator structure
mutable struct BatchEstimator
    time_steps::Vector{Float32}
    num_landmarks::Int32

    BatchEstimator(data::Dict) = create_batch_estimator(new(), data)
end

# Define batch estimator functions
function create_batch_estimator(estimator::BatchEstimator, data::Dict)
    times = data["t"]
    estimator.time_steps = [
        times[1, k] - times[1, k-1] for k in 2:size(times, 2)
    ]
    estimator.num_landmarks = 20
    return estimator
end


# function forward(batch_estimator::BatchEstimator)


# end