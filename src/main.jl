using Revise
using MAT

# include("utils.jl")
include("estimator.jl")

function main()
    # Load raw data
    data = matread("data.mat")

    # Create Batch Estimator from data
    batch_estimator = BatchEstimator(data)

    # Do dead reconing
    println("Working")
    dead_reconing!(batch_estimator)
end

main()
