using MAT

include("estimator.jl")

# Load raw data
data = matread("data.mat")

function main()

    # Create Batch Estimator from data
    batch_estimator = BatchEstimator(data)

    # Do dead reconings
    println("Working")
    dead_reconing!(batch_estimator)
    plot(batch_estimator)
end

main()