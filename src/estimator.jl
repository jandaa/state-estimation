using CoordinateTransformations, Rotations, StaticArrays
using LinearAlgebra
using TransformUtils
using Plots
using PyPlot;
const plt = PyPlot;
pygui(true)

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
    states::Array{Float32,3}

    # Ground truth
    ground_truth::Array{Float32,3}

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

    return estimator
end

function ground_truth_k(estimator::BatchEstimator, k)
    C = so3_exp(estimator.θ_vₖ_i[:, k])
    d = estimator.rᵢ_vₖ_i[:, k]
    return [C -C*d; 0 0 0 1]
end


function forward(estimator::BatchEstimator, k::Int64, T::Matrix{<:AbstractFloat})
    # Rotation = Angular velocity * time period
    psi = estimator.ω_vₖ_vₖ_i[:, k] * estimator.time_steps[k]

    # Distance = Linear velocity * time period
    d = estimator.v_vₖ_vₖ_i[:, k] * estimator.time_steps[k]

    # Change in pose
    dT = inv(se3_exp(d, -psi))

    return dT * T
end

function dead_reconing!(estimator::BatchEstimator)
    k1 = 1216
    k2 = 1714

    batch_size = k2 - k1

    # Get initial state from ground truth
    curr_state = ground_truth_k(estimator, k1)

    # Set initial state
    estimator.states = Array{Float32,3}(undef, 4, 4, batch_size)
    estimator.states[:, :, 1] = curr_state

    # Set ground truth
    estimator.ground_truth = Array{Float32,3}(undef, 4, 4, batch_size)
    estimator.ground_truth[:, :, 1] = inv(curr_state)


    for k ∈ k1:k2-1
        curr_state = forward(estimator, k, curr_state)
        estimator.states[:, :, k-k1+1] = curr_state

        estimator.ground_truth[:, :, k-k1+1] = inv(ground_truth_k(estimator, k))
    end
end

function plot(estimator::BatchEstimator)
    num_steps = size(estimator.states, 3)

    # Create plot
    ax = plt.axes(projection = "3d")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    states = [inv(estimator.states[:, :, k]) for k ∈ 1:num_steps]

    # Get state estimates
    x = [states[k][1, 4] for k ∈ 1:num_steps]
    y = [states[k][2, 4] for k ∈ 1:num_steps]
    z = [states[k][3, 4] for k ∈ 1:num_steps]
    ax.plot3D(x, y, z, c = "red", label = "Dead Reconing")

    # Get ground truth
    gt_x = [estimator.ground_truth[1, 4, k] for k ∈ 1:num_steps]
    gt_y = [estimator.ground_truth[2, 4, k] for k ∈ 1:num_steps]
    gt_z = [estimator.ground_truth[3, 4, k] for k ∈ 1:num_steps]
    ax.plot3D(gt_x, gt_y, gt_z, c = "blue", label = "GT")

    # Landmark positions
    l_x = [estimator.rho_i_pj_i[1, i] for i ∈ 1:estimator.num_landmarks]
    l_y = [estimator.rho_i_pj_i[2, i] for i ∈ 1:estimator.num_landmarks]
    l_z = [estimator.rho_i_pj_i[3, i] for i ∈ 1:estimator.num_landmarks]
    ax.scatter(l_x, l_y, l_z, c = "r", label = "landmarks")

    title("Estimate vs Ground Truth")
    plt.legend()
    plt.show()

end