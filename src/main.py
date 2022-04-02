from liegroups import SE3, SO3
from scipy.io import loadmat
from math import cos, sin, sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.linalg import block_diag

D = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
)


def skew(r):
    return np.array(
        [
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0],
        ]
    )


def so3_exp(psi):

    angle = np.linalg.norm(psi)

    axis = psi / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) - s * skew(axis)


def so3_log(C):

    cos_angle = 0.5 * np.trace(C) - 0.5

    # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if np.isclose(angle, 0.0):
        tmp = C - np.eye(3)

    else:
        tmp = (angle / (2 * np.sin(angle))) * (C - C.T)

    return np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])


def so3_jacob(psi):
    angle = np.linalg.norm(psi)

    if np.isclose(angle, 0.0):
        return np.eye(3) + 0.5 * skew(psi)

    axis = psi / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return (
        (s / angle) * np.eye(3)
        + (1 - s / angle) * np.outer(axis, axis)
        + (1 - c) / angle * skew(axis)
    )


def se3_exp(d, psi):

    # Compute jacobian
    J = so3_jacob(psi)

    C = so3_exp(psi)
    return np.block([[C, (J @ d).reshape(3, 1)], [0, 0, 0, 1]])


def se3_log(T):
    psi = so3_log(T[0:3, 0:3])

    J = so3_jacob(psi)
    r = np.linalg.inv(J) @ T[0:3, 3]

    return np.hstack((r, psi))


class BatchEstimator:
    """
    Batch Estimator
    """

    def __init__(self, dataset):
        """ """

        # Load all input data
        dataset = loadmat(dataset)

        # Extract scalar values that are stored in 2D arrays
        for key in dataset:
            if key.startswith("__"):
                continue
            if dataset[key].size == 1:
                dataset[key] = dataset[key][0][0]
            setattr(self, key, dataset[key])

        self.rho_v_c_v = self.rho_v_c_v.reshape(
            -1,
        )
        self.v_var = (
            self.v_var.reshape(
                -1,
            )
            * 3
        )
        self.w_var = self.w_var.reshape(
            -1,
        )
        self.y_var = self.y_var.reshape(
            -1,
        )

        # Time steps
        self.time_steps = [
            self.t[:, k] - self.t[:, k - 1] for k in range(1, len(self.t[0]))
        ]

        # Number of landmark points in the map
        self.num_points = 20

        self.T_c_v = np.block(
            [[self.C_c_v, -self.C_c_v @ self.rho_v_c_v.reshape(3, 1)], [0, 0, 0, 1]]
        )

    def ground_truth_k(self, k):
        C_vk_i = so3_exp(self.theta_vk_i[:, k])
        # C_vk_i = SO3.exp(self.theta_vk_i[:, k]).as_matrix()
        T = SE3.from_matrix(
            np.block(
                [[C_vk_i, -C_vk_i @ self.r_i_vk_i[:, k].reshape(3, 1)], [0, 0, 0, 1]]
            ),
            normalize=True,
        )
        return T

    def ground_truth(self, k1=1215, k2=1714):
        """
        return the ground truth poses.
        """
        # theta_vk_i is in axis-angle representation
        # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotating_a_vector
        return [self.ground_truth_k(k) for k in range(k1, k2 + 1)]

    def forward(self, T_prev, psi, d):
        """Forward Dead Reckoning step."""
        dT = SE3.from_matrix(se3_exp(d, -psi)).inv()
        return dT.dot(T_prev)

    def dead_reckoning(self, k1=1215, k2=1714, init=None):
        """
        Estimate position using only dead reckoning
        from the input velocities
        """
        states = []

        # Set initial state using ground-truth values
        if init:
            states.append(init)
        else:
            states.append(self.ground_truth_k(k1))

        # For each time step, computes the difference in
        # rotation and position due to the linear and angular
        # velocities.
        for k in range(k1, k2):

            # Rotation = Angular velocity * time period
            psi = self.w_vk_vk_i[:, k] * self.time_steps[k]

            # Distance = Linear velocity * time period
            d = self.v_vk_vk_i[:, k] * self.time_steps[k]

            states.append(self.forward(states[-1], psi, d))

        return states

    # Given a point p in the camera frame, project it
    # on the images such that pixels are returned
    def g(self, p):
        """Given a point and pose project into camera."""

        return np.array(
            [
                self.fu * p[0] / p[2] + self.cu,
                self.fv * p[1] / p[2] + self.cv,
                self.fu * (p[0] - self.b) / p[2] + self.cu,
                self.fv * p[1] / p[2] + self.cv,
            ]
        )

    # Given the state {r,C} of the vehicle, project the
    # i-th point expressed wrt the inertial frame
    # into the camera frame and return its [x,y,z] coords.
    def p_i(self, T, i):
        """Project point into camera frame."""
        return D.T @ self.T_c_v @ T.as_matrix() @ np.block([self.rho_i_pj_i[:, i], 1])

    # Jacobian of g() wrt to point p
    def dg_dp(self, p):
        """Camera jacobian."""
        x = p[0]
        y = p[1]
        z = p[2]
        z_squared = z ** 2
        dg_dp = np.array(
            [
                [self.fu / z, 0, -self.fu * x / z_squared],
                [0, self.fv / z, -self.fv * y / z_squared],
                [self.fu / z, 0, -self.fu * (x - self.b) / z_squared],
                [0, self.fv / z, -self.fv * y / z_squared],
            ]
        )
        return dg_dp

    def dg_dp_numerical(self, P):
        """Camera jacobian."""
        d = 0.00001
        dp_dx = self.g(P + np.array([d, 0, 0])) - self.g(P)
        dp_dy = self.g(P + np.array([0, d, 0])) - self.g(P)
        dp_dz = self.g(P + np.array([0, 0, d])) - self.g(P)

        return np.block(
            [dp_dx.reshape(4, 1) / d, dp_dy.reshape(4, 1) / d, dp_dz.reshape(4, 1) / d]
        )

    def dp_dx(self, T, p):
        """
        C: Orientation of the vehicle wrt inertial frame
        r: Position of the vehicle wrt inertial frame, expressed in inertial frame
        p: Position of the observed point wrt inertial frame
        """
        return D.T @ self.T_c_v @ SE3.odot(T.as_matrix() @ np.block([p, 1]))

    # Jacobian of the transformation that brings a point on the floor
    # to a position in the camera frame. This transformation is p_i()
    def dp_dx_numerical(self, T, p):
        """
        C: Orientation of the vehicle wrt inertial frame
        r: Position of the vehicle wrt inertial frame, expressed in inertial frame
        p: Position of the observed point wrt inertial frame
        """
        p = np.block([p, 1])
        T = T.as_matrix()
        d = 0.00001
        dp_de1 = D.T @ (
            SE3.exp(np.array([d, 0, 0, 0, 0, 0])).as_matrix() @ T @ p - T @ p
        )
        dp_de2 = D.T @ (
            SE3.exp(np.array([0, d, 0, 0, 0, 0])).as_matrix() @ T @ p - T @ p
        )
        dp_de3 = D.T @ (
            SE3.exp(np.array([0, 0, d, 0, 0, 0])).as_matrix() @ T @ p - T @ p
        )
        dp_de4 = D.T @ (
            SE3.exp(np.array([0, 0, 0, d, 0, 0])).as_matrix() @ T @ p - T @ p
        )
        dp_de5 = D.T @ (
            SE3.exp(np.array([0, 0, 0, 0, d, 0])).as_matrix() @ T @ p - T @ p
        )
        dp_de6 = D.T @ (
            SE3.exp(np.array([0, 0, 0, 0, 0, d])).as_matrix() @ T @ p - T @ p
        )

        return np.block(
            [
                dp_de1.reshape(3, 1) / d,
                dp_de2.reshape(3, 1) / d,
                dp_de3.reshape(3, 1) / d,
                dp_de4.reshape(3, 1) / d,
                dp_de5.reshape(3, 1) / d,
                dp_de6.reshape(3, 1) / d,
            ]
        )

    # Jacobian of g() wrt to the perturbations given
    # current pose {C,r} and observed point p
    def dg_dx(self, T, p):
        # Perform chain-rule
        return self.dg_dp(p) @ self.dp_dx(T, p)

    # Jacobian of the motion model wrt the perturbation
    # If you have the previous pose and the current pose,
    # this can be computed as Ad( T_curr inv(T_prev) ).
    # The provided poses needs to be transformation matrices.
    def motion_jacobian(self, T_prev, T_curr):
        F = SE3.adjoint(T_curr.dot(T_prev.inv()))
        return F

    # Return the error between two poses expressed as
    # transformation matrices. The error is a 6x1 vector.
    def pose_error(self, pose_1, pose_2):
        error = SE3.log(pose_1.dot(pose_2.inv()))
        return error

    # Return the error, in pixels between an observation
    # and the predicted pixel position (4x1) of a landmark
    def measurement_error(self, obs_1, obs_2):
        return obs_1 - obs_2

    def measurement_to_point(self, p):
        ul = p[0]
        vl = p[1]
        ur = p[2]
        vr = p[3]
        z = self.fu * self.b / (ul - ur)
        x = (ul - self.cu) * z / self.fu
        y = (vl - self.cv) * z / self.fv
        return np.array([x, y, z, 1])

    def verify_projections(self, k1=1215, k2=1714):

        ground_truth = self.ground_truth(k1=k1, k2=k2)

        for k in range(k1, k2):
            T = ground_truth[k - k1]
            C = T.rot.as_matrix()
            r = T.trans
            for j in range(self.num_points):
                if np.all(self.y_k_j[:, k, j] != np.array([-1, -1, -1, -1])):
                    P = self.p_i(T, j)

                    # Error terms
                    error = self.y_k_j[:, k, j] - self.g(P)
                    print(error)

    def gauss_newton(self, k1=1215, k2=1714, max_iterations=8, init=None):
        """
        Estimate position using only dead reckoning
        from the input velocities
        """

        # Initalize the operating trajectory by using the
        # ground-truth for the first pose and then dead-reckoning
        # from there until the end of the trajectory.
        dead_reckoning = self.dead_reckoning(k1=k1, k2=k2, init=init)
        T0 = dead_reckoning[0]
        T_op = dead_reckoning

        # Number of points in batch
        batch_size = (k2 - k1) + 1

        # Holds the perturbations used on each iteration
        delta_x = np.zeros((batch_size, 6))

        # Width of H jacobian matrix
        width = 6 * batch_size

        # Iterate refinement until convergence
        for iteration in range(max_iterations):

            # Initialize lists of arrays
            ev = []  # Motion errors throughout trajectory
            ey = []  # Measurement errors for observations
            Gs = []  # Lower part of the H jacobian matrix
            Fs = []  # Upper part of the H jacobian matrix

            # Variance P0 of the initial pose
            # TODO: Why 0.2 ?
            sigma_init = 1  # 0.2
            # Wv_inv is the inverse of the first part of the W matrix
            # with diagonal entries related to process noise.
            Wv_inv = 6 * [sigma_init]
            # Wy_inv is the inverse of the second part of the W matrix
            # with diagonal entries related to measurement noise.
            Wy_inv = []

            # Initialize the motion error list
            ev = list(self.pose_error(T0, T_op[0]).flatten())

            # Each "row" of Fs consists in a 6 x 6K array
            # The first one has only np.eye(6) at the front of the row
            F = np.zeros((6, width))
            F[:, 0:6] = np.eye(6)
            Fs.append(F)

            # This loop generates the upper part of the H jacobian matrix
            # and the upper part of the error matrix.
            for k in range(k1, k2):
                # Relative k index starting at 0 when k = k1
                k_rel = k - k1

                # Current pose T_curr and previous pose T_prev
                T_curr = T_op[k_rel + 1]
                T_prev = T_op[k_rel]

                # Angular perturbation = Angular velocity * time period
                psi = self.w_vk_vk_i[:, k] * self.time_steps[k]

                # Linear perturbation = Linear velocity * time period
                d = self.v_vk_vk_i[:, k] * self.time_steps[k]

                # Given the previous pose, use the motion model to predict
                # the current pose.
                T_pred = self.forward(T_prev, psi, d)
                # Compare predicted pose with current pose
                error = self.pose_error(T_pred, T_curr)
                # Append the error to the list
                ev += list(error.flatten())

                # Since the Q matrix is diagonal, the inverse of this matrix
                # is obtained by taking the reciprocal of the diagonal elements.
                # The variance of the linear and angular process noise must be multiplied
                # by the squared period of the timestep since the variance is expressed
                # in [m/s] and [rad/s] but we want [m] and [rad].
                qr_inv = list(1 / (self.v_var * (self.time_steps[k] ** 2)))
                qc_inv = list(1 / (self.w_var * (self.time_steps[k] ** 2)))
                q_inv = qr_inv + qc_inv
                # Append the inverse of the Q matrix to the list
                Wv_inv += q_inv

                # Motion jacobian of the previous timestep F_{k-1}
                F = self.motion_jacobian(T_prev, T_curr)

                # Most of the elements of the row are zeros
                F_row = np.zeros((6, width))
                # Horizontal offset where to place the -1*F matrix
                offset = 6 * k_rel
                F_row[:, offset : offset + 6] = -1 * F
                # Horizontal offset where to place the identity matrix
                offset = offset + 6
                F_row[:, offset : offset + 6] = np.eye(6)

                # Append the produced row to the list
                Fs.append(F_row)

            # This loop generates the lower part of the H jacobian matrix
            # and the lower part of the error matrix.
            for k in range(k1, k2 + 1):
                # Produced matrices for the k-th timestep
                Gk = []  # Measurement jacobian
                eyk = []  # Observation errors
                Rk_inv = []  # Inverted measurement covariance matrix

                # Relative k index starting at 0 when k = k1
                k_rel = k - k1

                # For convenience
                T = T_op[k_rel]
                C = T.rot.as_matrix()
                r = T.trans

                # Iterate through all of the possible landmarks
                for j in range(self.num_points):
                    # Pixels values (4x1) of the j-th landmark
                    # observed at the k-th timestep.
                    landmark = self.y_k_j[:, k, j]

                    # If the landmark is invalid, each pixel will be set to -1
                    if landmark[0] > 0:

                        # Project the j-th point in camera frame
                        P_cam = self.p_i(T, j)
                        # Project a point in space to a point in images
                        P_image = self.g(P_cam)

                        # Compute the measurement error and append to list
                        meas_error = self.measurement_error(landmark, P_image)
                        error = list(meas_error.flatten())
                        eyk += error

                        # Append the inverse of R
                        Rk_inv += list(1 / self.y_var)

                        # Jacobian of g() wrt perturbations given pose and landmark
                        # G = self.dg_dx(T, self.rho_i_pj_i[:,j])
                        G = self.dg_dp(P_cam) @ self.dp_dx(T, self.rho_i_pj_i[:, j])
                        Gk.append(G)

                # Add the optimization variables to the problem
                # if you were able to observe more than a certain
                # number of landmarks.
                if len(Gk) > 0:
                    ey += eyk
                    Wy_inv += Rk_inv

                    # Each G jacobian is a 4x6 matrix
                    G_row = np.zeros((4 * len(Gk), width))
                    # Offset each jacobian below and right of the previous one
                    for i, Gki in enumerate(Gk):
                        vert_offset = 4 * i  # Vertical offset
                        horz_offset = 6 * k_rel  # Horizontal offset
                        G_row[
                            vert_offset : vert_offset + 4, horz_offset : horz_offset + 6
                        ] = Gki
                    Gs.append(G_row)

            # Finally, the H matrix can be produced from its upper and lower parts
            H = np.vstack(tuple(Fs + Gs))
            # Build the inverse covariance matrix
            W_inv = np.diag(tuple(Wv_inv + Wy_inv))
            # Build the error matrix
            e = np.array(ev + ey)

            # For Ax=b and solving for x
            A = H.T @ W_inv @ H
            b = H.T @ W_inv @ e

            # We solve for the optimal perturbations \delta_x
            # for the whole trajectory. Each perturbation is a
            # 6x1 vector.
            x_opt = np.linalg.solve(A, b).flatten()

            # Reshape the very 6*N long vector in a Nx6 matrix where
            # N is the number of timesteps in the trajectory.
            delta_x = np.reshape(x_opt, (len(T_op), 6))

            # To update the operating point of the optimization we
            # need to unstack all optimal perturbations and then
            # use T = exp(x^)T to update the operating point T
            # which represents our best guess in terms of pose
            norms = []
            for k in range(len(T_op)):

                T_op[k] = SE3.exp(delta_x[k]).dot(T_op[k])

                l2_norm = np.linalg.norm(delta_x[k])
                norms = norms + [l2_norm]
            print("Average L2 norm of perturbations: {}".format(np.mean(norms)))

            if np.mean(norms) < 1e-2:
                break

        # A is the inverse covariance matrix
        return T_op, np.linalg.inv(A).diagonal().reshape(batch_size, 6)


def plot3d(states):
    """
    Plot the 3D trajectory
    """

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    rs = [state.trans for state in states]
    x = [r[0] for r in rs]
    y = [r[1] for r in rs]
    z = [r[2] for r in rs]

    ax.plot3D(x, y, z)
    plt.show()


def plot3d_compare(data_dict, scatter_data_dict):
    """
    Plot the 3D trajectory
    """

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for data in data_dict:

        rs = [state.trans for state in data_dict[data]]
        x = [r[0] for r in rs]
        y = [r[1] for r in rs]
        z = [r[2] for r in rs]

        ax.plot3D(x, y, z, label=data)

    for data in scatter_data_dict:

        x = [r[0] for r in scatter_data_dict[data]]
        y = [r[1] for r in scatter_data_dict[data]]
        z = [r[2] for r in scatter_data_dict[data]]
        ax.scatter(x, y, z, label=data)

    plt.legend()
    plt.show()


def plot_errors_batch(estimates, cov, ground_truth, prefix):

    trans_errors = [
        T_star.trans - T.trans for T_star, T in zip(estimates, ground_truth)
    ]

    rot_errros = [
        SO3.vee(np.eye(3) - T_star.rot.dot(T.rot.inv()).as_matrix())
        for T_star, T in zip(estimates, ground_truth)
    ]

    # Plot translation errors
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

    ax1.set_title("Translation Error: x")
    ax1.set_ylabel("Error (m)")
    ax1.plot([e[0] for e in trans_errors])
    ax1.plot([3 * sqrt(cov[i][0]) for i in range(len(trans_errors))], ":r")
    ax1.plot([-3 * sqrt(cov[i][0]) for i in range(len(trans_errors))], ":r")

    ax2.set_title("Translation Error: y")
    ax2.set_ylabel("Error (m)")
    ax2.plot([e[1] for e in trans_errors])
    ax2.plot([3 * sqrt(cov[i][1]) for i in range(len(trans_errors))], ":r")
    ax2.plot([-3 * sqrt(cov[i][1]) for i in range(len(trans_errors))], ":r")

    ax3.set_title("Translation Error: z")
    ax3.set_ylabel("Error (m)")
    ax3.set_xlabel("tk")
    ax3.plot([e[2] for e in trans_errors])
    ax3.plot([3 * sqrt(cov[i][2]) for i in range(len(trans_errors))], ":r")
    ax3.plot([-3 * sqrt(cov[i][2]) for i in range(len(trans_errors))], ":r")

    fig.savefig(prefix + "_trans_errors.pdf")

    # Plot rotational errors
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

    ax1.set_title("Rotational Error: x")
    ax1.set_ylabel("Error (rad)")
    ax1.plot([e[0] for e in rot_errros])
    ax1.plot([3 * sqrt(cov[i][3]) for i in range(len(trans_errors))], ":r")
    ax1.plot([-3 * sqrt(cov[i][3]) for i in range(len(trans_errors))], ":r")

    ax2.set_title("Rotational Error: y")
    ax2.set_ylabel("Error (rad)")
    ax2.plot([e[1] for e in rot_errros])
    ax2.plot([3 * sqrt(cov[i][4]) for i in range(len(trans_errors))], ":r")
    ax2.plot([-3 * sqrt(cov[i][4]) for i in range(len(trans_errors))], ":r")

    ax3.set_title("Rotational Error: z")
    ax3.set_xlabel("tk")
    ax3.set_ylabel("Error (rad)")
    ax3.plot([e[2] for e in rot_errros])
    ax3.plot([3 * sqrt(cov[i][5]) for i in range(len(trans_errors))], ":r")
    ax3.plot([-3 * sqrt(cov[i][5]) for i in range(len(trans_errors))], ":r")

    fig.savefig(prefix + "_rot_errors.pdf")


# Question 5
estimator = BatchEstimator("data.mat")

# # Part 1:
# k1 = 1215
# k2 = 1714
# ground_truth = estimator.ground_truth(k1=k1, k2=k2)
# gauss_newton, cov = estimator.gauss_newton(k1=k1, k2=k2)
# plot_errors_batch(gauss_newton, cov, ground_truth, "part1")

# # Part 2
# window_size = 50
# poses = []
# covariences = []
# for k in range(k1, k2):
#     if k == k1:
#         init = estimator.ground_truth_k(k1)
#     else:
#         init = poses[-1]

#     gauss_newton, cov = estimator.gauss_newton(k1=k, k2=k + window_size, init=init)

#     poses.append(gauss_newton[0])
#     covariences.append(cov[0])

# covariences = np.array(covariences)
# plot_errors_batch(poses, covariences, ground_truth, "part2")

# window_size = 10
# poses = []
# covariences = []
# for k in range(k1, k2):
#     if k == k1:
#         init = estimator.ground_truth_k(k1)
#     else:
#         init = poses[-1]

#     gauss_newton, cov = estimator.gauss_newton(k1=k, k2=k + window_size, init=init)

#     poses.append(gauss_newton[0])
#     covariences.append(cov[0])

# covariences = np.array(covariences)
# plot_errors_batch(poses, covariences, ground_truth, "part3")


ground_truth = estimator.ground_truth()
dead_reckoning = estimator.dead_reckoning()
points = [estimator.rho_i_pj_i[:, i] for i in range(20)]
# gauss_newton = estimator.gauss_newton()

# estimator.verify_projections()

# Test point cloud allignment
# point_cloud_allignment = estimator.point_cloud_allignment(num_iterations=10)
gauss_newton, _ = estimator.gauss_newton(max_iterations=8)
plot3d_compare(
    {
        "Ground Truth": [T.inv() for T in ground_truth],
        "Dead Reckoning": [T.inv() for T in dead_reckoning],
        "Gauss Newton": [T.inv() for T in gauss_newton],
    },
    {
        "Points": points,
        "Intertial Frame": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        # "Point Cloud Allignment": [point_cloud_allignment[i].inv().trans for i in range(len(point_cloud_allignment))],
    },
)

# # Test Gauss Newton
# dead_reckoning = estimator.dead_reckoning()
# gauss_newton = estimator.gauss_newton(num_iterations=8)
# plot3d_compare(
#     {
#         "Dead Reckoning": dead_reckoning,
#         "Ground Truth": ground_truth,
#         "Gauss Newton": gauss_newton,
#     },
#     {
#         "Points": points
#     }
# )
