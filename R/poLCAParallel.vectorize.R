#' poLCAParallel - Vectorize
#' Mimics poLCA.vectorize but with some of the dimensions swapped
#'
#' Given a list of probabilities of outcomes, for each cluster, flatten it into
    #' a vector, suitable for EmAlgorithmRcpp
#' @param probs list of length n_category, for the ith entry, it contains a
    #' matrix of outcome probabilities with dimensions n_class x n_outcomes[i]
#' @return a list containing:
    #' vecprobs: vector of outcome probabilities, a flatten list of matrices
        #' dim 0: for each outcome
        #' dim 1: for each category
        #' dim 2: for each cluster
    #' numChoices: vector, number of outcomes for each category
poLCAParallel.vectorize = function(probs) {
    classes = nrow(probs[[1]])
    vecprobs = c()
    for (m in 1: classes) {
        for (j in 1: length(probs)) {
            vecprobs = c(vecprobs, probs[[j]][m, ])
        }
    }
    num_choices = sapply(probs, ncol)
    return(list(vecprobs=vecprobs, numChoices=num_choices, classes=classes))
}
