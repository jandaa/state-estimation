using CoordinateTransformations, Rotations, StaticArrays
using LinearAlgebra
using TransformUtils
using Plots
using PyPlot;
const plt = PyPlot;
pygui(true)

include("lie_groups.jl")

D = [
    1 0 0
    0 1 0
    0 0 1
    0 0 0
]

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

    # Transformation from vehicle to camera frame
    T_c_v::Matrix{Float32}

    # Estimated states for each timestep
    states::Array{Float32,3}

    # Ground truth
    ground_truth::Vector{Matrix{Float32}}

    # Variances
    v_var::Matrix{Float32}
    w_var::Matrix{Float32}
    y_var::Matrix{Float32}

    # Scalars
    num_landmarks::Int
    fu::Float32
    fv::Float32
    cu::Float32
    cv::Float32
    b::Float32
    k1::Int
    k2::Int
    batch_size::Int
    max_iterations::Int

    BatchEstimator(data::Dict) = create_batch_estimator!(new(), data)
end

# Define batch estimator functions
function create_batch_estimator!(estimator::BatchEstimator, data::Dict)

    # Extract raw data
    estimator.ω_vₖ_vₖ_i = data["w_vk_vk_i"]
    estimator.v_vₖ_vₖ_i = data["v_vk_vk_i"]
    estimator.θ_vₖ_i = data["theta_vk_i"]
    estimator.rᵢ_vₖ_i = data["r_i_vk_i"]
    estimator.rho_i_pj_i = data["rho_i_pj_i"]
    estimator.yₖ_j = data["y_k_j"]

    # Variances
    estimator.v_var = data["v_var"] * 3
    estimator.w_var = data["w_var"]
    estimator.y_var = data["y_var"]

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

    # Set vehicle to camera transform
    estimator.T_c_v = [data["C_c_v"] -data["C_c_v"]*data["rho_v_c_v"]; 0 0 0 1]

    # Set the period we're looking at 
    estimator.k1 = 1218
    estimator.k2 = 1426
    estimator.batch_size = estimator.k2 - estimator.k1

    # Set ground truth
    estimator.ground_truth = ground_truth(estimator)

    # Set gauss newton parameters
    estimator.max_iterations = 3

    return estimator
end

function ground_truth_k(estimator::BatchEstimator, k)
    C = so3_exp(estimator.θ_vₖ_i[:, k])
    d = estimator.rᵢ_vₖ_i[:, k]
    return [C -C*d; 0 0 0 1]
end

function ground_truth(estimator::BatchEstimator)
    return [
        inv(ground_truth_k(estimator, k))
        for k ∈ estimator.k1:estimator.k2
    ]
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

function dead_reconing(estimator::BatchEstimator)

    # Get initial state from ground truth
    curr_state = ground_truth_k(estimator, estimator.k1)

    # Set initial state
    states = Array{Float32,3}(undef, estimator.batch_size, 4, 4)
    states[1, :, :] = curr_state

    for k ∈ estimator.k1:estimator.k2-2

        # Compute next state
        curr_state = forward(estimator, k, curr_state)

        # update current state
        states[k-estimator.k1+2, :, :] = curr_state
    end

    return states
end

# Given a point p in the camera frame, project it
# on the images such that pixels are returned
function g(estimator::BatchEstimator, p::Vector{<:AbstractFloat})
    """Given a point and pose project into camera."""
    return [
        estimator.fu * p[1] / p[3] + estimator.cu
        estimator.fv * p[2] / p[3] + estimator.cv
        estimator.fu * (p[1] - estimator.b) / p[3] + estimator.cu
        estimator.fv * p[2] / p[3] + estimator.cv
    ]
end

# Given the state {r,C} of the vehicle, project the
# i-th point expressed wrt the inertial frame
# into the camera frame and return its [x,y,z] coords.
function p_i(estimator::BatchEstimator, T::Matrix{<:AbstractFloat}, i::Int64)
    """Project point into camera frame."""
    return D' * estimator.T_c_v * T * [estimator.rho_i_pj_i[:, i]; 1]
end

# Jacobian of the motion model wrt the perturbation
# If you have the previous pose and the current pose,
# this can be computed as Ad( T_curr inv(T_prev) ).
# The provided poses needs to be transformation matrices.
function motion_jacobian(T_prev::Matrix{<:AbstractFloat}, T_curr::Matrix{<:AbstractFloat})
    return se3_adjoint(T_curr * se3_inv(T_prev))
end

# Return the error between two poses expressed as
# transformation matrices. The error is a 6x1 vector.
function pose_error(T1::Matrix{<:AbstractFloat}, T2::Matrix{<:AbstractFloat})
    error = se3_log(T1 * se3_inv(T2))
    error[error.<1e-4] .= 0.0
    return error
end

# Return the error, in pixels between an observation
# and the predicted pixel position (4x1) of a landmark
function measurement_error(obs_1::Vector{<:AbstractFloat}, obs_2::Vector{<:AbstractFloat})
    return obs_1 - obs_2
end

function measurement_to_point(self, p)
    ul = p[0]
    vl = p[1]
    ur = p[2]
    vr = p[3]
    z = self.fu * self.b / (ul - ur)
    x = (ul - self.cu) * z / self.fu
    y = (vl - self.cv) * z / self.fv
    return np.array([x, y, z, 1])
end

# Jacobian of g() wrt to point p
function dg_dp(estimator::BatchEstimator, p::Vector{<:AbstractFloat})
    """Camera jacobian."""
    x = p[1]
    y = p[2]
    z = p[3]
    z_squared = z^2
    return [
        estimator.fu/z 0 -estimator.fu*x/z_squared
        0 estimator.fv/z -estimator.fv*y/z_squared
        estimator.fu/z 0 -estimator.fu*(x-estimator.b)/z_squared
        0 estimator.fv/z -estimator.fv*y/z_squared
    ]
end

function dp_dx(estimator::BatchEstimator, T::Matrix{<:AbstractFloat}, p::Vector{<:AbstractFloat})
    """
    C: Orientation of the vehicle wrt inertial frame
    r: Position of the vehicle wrt inertial frame, expressed in inertial frame
    p: Position of the observed point wrt inertial frame
    """
    return D' * estimator.T_c_v * se3_odot(T * [p; 1])
end

function gauss_newton(estimator::BatchEstimator, initial_estimate::Array{<:AbstractFloat,3})

    # Initalize guass newton updated states
    T_op = copy(initial_estimate)
    T0 = T_op[1, :, :]

    # Holds the perturbations used on each iteration
    delta_x = zeros(estimator.batch_size, 6)

    # Width of H jacobian matrix
    width = 6 * estimator.batch_size

    # Iterate refinement until convergence
    for i ∈ 1:estimator.max_iterations

        # Initialize lists of arrays
        ev = Vector{Float32}()  # Motion errors throughout trajectory
        ey = Vector{Float32}()          # Measurement errors for observations
        Gs = Vector{Matrix{Float32}}()  # Lower part of the H jacobian matrix

        # Variance P0 of the initial pose
        sigma_init = 1

        # Wv_inv is the inverse of the first part of the W matrix
        # with diagonal entries related to process noise.
        Wv_inv = ones(6, 1) * sigma_init

        # Wy_inv is the inverse of the second part of the W matrix
        # with diagonal entries related to measurement noise.
        Wy_inv = Vector{Matrix{Float32}}()

        # push!(ev, pose_error(T0, T_op[1, :, :]))
        ev = [ev; pose_error(T0, T_op[1, :, :])]

        # Each "row" of Fs consists in a 6 x 6K array
        # The first one has only np.eye(6) at the front of the row
        F = zeros(6, width)
        F[:, 1:6] = I(6)
        Fs = F

        # This loop generates the upper part of the H jacobian matrix
        # and the upper part of the error matrix.
        for k ∈ estimator.k1:estimator.k2-2

            # Relative k index starting at 0 when k = k1
            k_rel = k - estimator.k1 + 1

            # Current pose T_curr and previous pose T_prev
            T_curr = T_op[k_rel+1, :, :]
            T_prev = T_op[k_rel, :, :]

            # Given the previous pose, use the motion model to predict
            # the current pose.
            T_pred = forward(estimator, k, T_prev)

            # Compare predicted pose with current pose
            error = pose_error(T_pred, T_curr)
            # Append the error to the list
            # push!(ev, error)
            ev = [ev; error]

            # Since the Q matrix is diagonal, the inverse of this matrix
            # is obtained by taking the reciprocal of the diagonal elements.
            # The variance of the linear and angular process noise must be multiplied
            # by the squared period of the timestep since the variance is expressed
            # in [m/s] and [rad/s] but we want [m] and [rad].
            qr_inv = 1.0 ./ (estimator.v_var * (estimator.time_steps[k]^2))
            qc_inv = 1.0 ./ (estimator.w_var * (estimator.time_steps[k]^2))
            q_inv = [qr_inv; qc_inv]

            # Append the inverse of the Q matrix to the list
            Wv_inv = [Wv_inv; q_inv]

            # Motion jacobian of the previous timestep F_{k-1}
            F = motion_jacobian(T_prev, T_curr)

            # Most of the elements of the row are zeros
            F_row = zeros(6, width)
            # Horizontal offset where to place the -1*F matrix
            offset = 6 * (k_rel - 1) + 1
            F_row[:, offset:offset+5] = -1 * F
            # Horizontal offset where to place the identity matrix
            offset = offset + 6
            F_row[:, offset:offset+5] = I(6)

            # Append the produced row to the list
            Fs = vcat(Fs, F_row)
        end

        # This loop generates the lower part of the H jacobian matrix
        # and the lower part of the error matrix.
        for k ∈ estimator.k1:estimator.k2-1

            # Produced matrices for the k-th timestep
            Gk = Vector{Matrix{Float32}}()      # Measurement jacobian
            eyk = Vector{Float32}()     # Observation errors
            Rk_inv = Vector{Matrix{Float32}}()  # Inverted measurement covariance matrix

            # Relative k index starting at 0 when k = k1
            k_rel = k - estimator.k1 + 1

            # For convenience
            T = T_op[k_rel, :, :]

            # Iterate through all of the possible landmarks
            for j ∈ 1:estimator.num_landmarks

                # Pixels values (4x1) of the j-th landmark
                # observed at the k-th timestep.
                landmark = estimator.yₖ_j[:, k, j]

                # If the landmark is invalid, each pixel will be set to -1
                if landmark[1] > 0

                    # Project the j-th point in camera frame
                    P_cam = p_i(estimator, T, j)
                    # Project a point in space to a point in images
                    P_image = g(estimator, P_cam)

                    # Compute the measurement error and append to list
                    meas_error = measurement_error(landmark, P_image)
                    eyk = [eyk; meas_error]

                    # Append the inverse of R
                    # push!(Rk_inv, 1 ./ estimator.y_var)
                    if size(Rk_inv, 1) > 0
                        Rk_inv = vcat([Rk_inv; 1 ./ estimator.y_var])
                    else
                        Rk_inv = 1 ./ estimator.y_var
                    end

                    # Jacobian of g() wrt perturbations given pose and landmark
                    G = dg_dp(estimator, P_cam) * dp_dx(estimator, T, estimator.rho_i_pj_i[:, j])
                    push!(Gk, G)
                end
            end

            # Add the optimization variables to the problem
            # if you were able to observe more than a certain
            # number of landmarks.
            if size(Gk, 1) > 0

                ey = [ey; eyk]
                if size(Wy_inv, 1) > 0
                    Wy_inv = [Wy_inv; Rk_inv]
                else
                    Wy_inv = Rk_inv
                end

                # Each G jacobian is a 4x6 matrix
                G_row = zeros(4 * size(Gk, 1), width)

                # Offset each jacobian below and right of the previous one
                for (i, Gki) ∈ enumerate(Gk)
                    vert_offset = 4 * (i - 1) + 1       # Vertical offset
                    horz_offset = 6 * (k_rel - 1) + 1   # Horizontal offset
                    G_row[
                        vert_offset:vert_offset+3, horz_offset:horz_offset+5
                    ] = Gki
                end

                if size(Gs, 1) > 1
                    Gs = vcat(Gs, G_row)
                else
                    Gs = G_row
                end
            end
        end

        # Finally, the H matrix can be produced from its upper and lower parts
        H = [Fs; Gs]
        # Build the inverse covariance matrix
        W_inv = Diagonal([Wv_inv; Wy_inv][:, 1])
        # Build the error matrix
        e = [ev; ey]

        # For Ax=b and solving for x
        A = H' * W_inv * H
        b = H' * W_inv * e

        # We solve for the optimal perturbations \delta_x
        # for the whole trajectory. Each perturbation is a
        # 6x1 vector.
        x_opt = A \ b

        # Reshape the very 6*N long vector in a Nx6 matrix where
        # N is the number of timesteps in the trajectory.

        # TODO: FIGURE THIS OUT
        delta_x = reshape(x_opt, 6, estimator.batch_size)'

        # To update the operating point of the optimization we
        # need to unstack all optimal perturbations and then
        # use T = exp(x^)T to update the operating point T
        # which represents our best guess in terms of pose
        for k ∈ 1:estimator.batch_size
            d = delta_x[k, 1:3]
            psi = delta_x[k, 4:6]
            # d = clamp!(d, -0.2, 0.2)
            # psi = clamp!(psi, -0.2, 0.2)
            T_op[k, :, :] = se3_exp(d, -psi) * T_op[k, :, :]
        end

    end

    return T_op
end

function plot_trajectory(estimator::BatchEstimator, dead_reconing::Array{<:AbstractFloat,3}, gauss_newton::Array{<:AbstractFloat,3})
    num_steps = estimator.batch_size

    # Create plot
    ax = plt.axes(projection="3d")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    states = [inv(dead_reconing[k, :, :]) for k ∈ 1:num_steps]

    # Get state estimates
    x = [states[k][1, 4] for k ∈ 1:num_steps]
    y = [states[k][2, 4] for k ∈ 1:num_steps]
    z = [states[k][3, 4] for k ∈ 1:num_steps]
    ax.plot3D(x, y, z, c="red", label="Dead Reconing")

    states = [inv(gauss_newton[k, :, :]) for k ∈ 1:num_steps]

    # Get state estimates
    x = [states[k][1, 4] for k ∈ 1:num_steps]
    y = [states[k][2, 4] for k ∈ 1:num_steps]
    z = [states[k][3, 4] for k ∈ 1:num_steps]
    ax.plot3D(x, y, z, c="purple", label="Gauss Newton")

    # Get ground truth
    gt_x = [estimator.ground_truth[k][1, 4] for k ∈ 1:num_steps]
    gt_y = [estimator.ground_truth[k][2, 4] for k ∈ 1:num_steps]
    gt_z = [estimator.ground_truth[k][3, 4] for k ∈ 1:num_steps]
    ax.plot3D(gt_x, gt_y, gt_z, c="blue", label="GT")

    # Landmark positions
    l_x = [estimator.rho_i_pj_i[1, i] for i ∈ 1:estimator.num_landmarks]
    l_y = [estimator.rho_i_pj_i[2, i] for i ∈ 1:estimator.num_landmarks]
    l_z = [estimator.rho_i_pj_i[3, i] for i ∈ 1:estimator.num_landmarks]
    ax.scatter(l_x, l_y, l_z, c="r", label="landmarks")

    title("Estimate vs Ground Truth")
    plt.legend()
    plt.show()

end