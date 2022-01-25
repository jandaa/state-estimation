using Plots
using PyPlot;
const plt = PyPlot;

function plot_gt(data)
    positions_gt = data["r_i_vk_i"]
    landmarks_gt = data["rho_i_pj_i"]

    ax = plt.axes(projection = "3d")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Extract x,y,z vectors
    x = [positions_gt[1, i] for i = 1:size(positions_gt, 2)]
    y = [positions_gt[2, i] for i = 1:size(positions_gt, 2)]
    z = [positions_gt[3, i] for i = 1:size(positions_gt, 2)]
    ax.plot3D(x, y, z, label = "GT")

    # Landmark positions
    l_x = [landmarks_gt[1, i] for i = 1:size(landmarks_gt, 2)]
    l_y = [landmarks_gt[2, i] for i = 1:size(landmarks_gt, 2)]
    l_z = [landmarks_gt[3, i] for i = 1:size(landmarks_gt, 2)]
    ax.scatter(l_x, l_y, l_z, c = "r", label = "landmarks")

    title("Ground truth Position")
    plt.legend()
    plt.show()
end