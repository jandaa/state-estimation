using CoordinateTransformations, Rotations, StaticArrays
using LinearAlgebra
using TransformUtils

include("lie_groups.jl")

# Define batch estimator structure
mutable struct BatchEstimator

    # Vectors
    time_steps::Vector{Float32}

    # Rotational velocity vk_vk
    ω_vₖ_vₖ_i::Matrix{Float32}

    # Linear velocity vk_vk
    v_vₖ_vₖ_i::Matrix{Float32}

    # GT rotation (axis-angle)
    θ_vₖ_i::Matrix{Float32}

    # GT position
    rᵢ_vₖ_i::Matrix{Float32}

    # landmark position
    rho_i_pj_i::Matrix{Float32}

    # observations
    yₖ_j::Array{Float32,3}

    # Rotation from vehicle to camera frame
    C_c_v::Matrix{Float32}

    # Translation from vehicle to camera frame
    rho_v_c_v::Matrix{Float32}

    # Estimated states for each timestep
    states::Array{Float32,2}

    # Scalars
    num_landmarks::Int
    fu::Float32
    fv::Float32
    cu::Float32
    cv::Float32
    b::Float32

    BatchEstimator(data::Dict) = create_batch_estimator!(new(), data)
end

# function

# Define batch estimator functions
function create_batch_estimator!(estimator::BatchEstimator, data::Dict)

    # Extract raw data
    estimator.ω_vₖ_vₖ_i = data["w_vk_vk_i"]
    estimator.v_vₖ_vₖ_i = data["v_vk_vk_i"]
    estimator.θ_vₖ_i = data["theta_vk_i"]
    estimator.rᵢ_vₖ_i = data["r_i_vk_i"]
    estimator.rho_i_pj_i = data["rho_i_pj_i"]
    estimator.yₖ_j = data["y_k_j"]
    estimator.C_c_v = data["C_c_v"]
    estimator.rho_v_c_v = data["rho_v_c_v"]

    # Constants
    estimator.num_landmarks = 20
    estimator.fu = data["fu"]
    estimator.fv = data["fv"]
    estimator.cu = data["cu"]
    estimator.cv = data["cv"]
    estimator.b = data["b"]

    # Time steps
    times = data["t"]
    estimator.time_steps = [
        times[1, k] - times[1, k-1] for k in 2:size(times, 2)
    ]

    # Set the inital state as ground truth
    estimator.states = Array{Float32,2}(undef, 6, size(times, 2))
    estimator.states[:, 1] = [estimator.rᵢ_vₖ_i[:, 1]; estimator.θ_vₖ_i[:, 1]]

    return estimator
end


function forward(estimator::BatchEstimator, k::Int64)
    # Rotation = Angular velocity * time period
    psi = estimator.ω_vₖ_vₖ_i[:, k] * estimator.time_steps[k]

    # Distance = Linear velocity * time period
    d = estimator.v_vₖ_vₖ_i[:, k] * estimator.time_steps[k]

    # Change in pose
    dT = inv(se3_exp(d, -psi))

    curr_state = estimator.states[:, k]
    T = se3_exp(curr_state[1:3], -curr_state[4:6])

    new_T = dT * T
    # return Matrix{Float32}(undef, 6, 1)
    return se3_log(new_T)
end

function dead_reconing!(estimator::BatchEstimator)
    for k ∈ 1:size(estimator.time_steps, 1)
        next_state = forward(estimator, k)
        estimator.states[:, k+1] = next_state
    end
end