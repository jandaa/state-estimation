using MAT

include("estimator.jl")

# Load raw data
data = matread("data.mat")

function main()

    # Create Batch Estimator from data
    batch_estimator = BatchEstimator(data)

    # Perform state estimation
    states1 = dead_reconing(batch_estimator)
    states2 = gauss_newton(batch_estimator, states1)
    plot_trajectory(batch_estimator, states1, states2)
end

main()