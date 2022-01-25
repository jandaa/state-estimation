using MAT

include("utils.jl")
include("estimator.jl")

# Load raw data
data = matread("dataset3.mat")

# Plot ground truth
# plot_gt(data)

# Create Batch Estimator from data
batch_estimator = BatchEstimator(data)

# Do dead reconing
println("Working")