#' Put all outcome probabilities into a single vector
#'
#' Given a list of matrices of outcome probabilities, flatten it into a vector,
#' suitable for EmAlgorithmRcpp. It mimics poLCA.vectorize() but with some of
#' the dimensions swapped to improve cache efficiency in the C++ code.
#'
#' @param probs list of length n_category. For the ith entry, it contains a
#' matrix of outcome probabilities with dimensions n_class x n_outcomes[i]
#' @return a list containing:
#'   * vecprobs: vector of outcome probabilities, a flattened list of matrices
#'     * dim 0: for each outcome
#'     * dim 1: for each category
#'     * dim 2: for each cluster
#'     * in other words, imagine a nested loop, from outer to inner:
#'         * for each cluster, for each category, for each outcome
#'   * numChoices: vector, number of outcomes for each category
#'   * classes: integer, number of classes (or clusters)
poLCAParallel.vectorize <- function(probs) {
    classes <- nrow(probs[[1]])
    vecprobs <- c()
    for (m in seq_len(classes)) {
        for (j in seq_len(length(probs))) {
            vecprobs <- c(vecprobs, probs[[j]][m, ])
        }
    }
    num_choices <- sapply(probs, ncol)
    return(list(
        vecprobs = vecprobs, numChoices = num_choices, classes = classes
    ))
}
