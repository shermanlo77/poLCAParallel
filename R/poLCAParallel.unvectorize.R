#' Reverse the operations of poLCAParallel.vectorize()
#'
#' Mimics poLCA.unvectorize but with some of the dimensions swapped. Given the
#' return value, or even modified, of poLCAParallel.vectorize(), return a list
#' of matrices containing outcome probabilities.
#'
#' @param vp list of three items (vecprobs, numChoices, classes) where
#'   * vecprobs: vector of outcome probabilities, a flattened list of matrices
#'      * dim 0: for each outcome
#'      * dim 1: for each category
#'      * dim 2: for each cluster
#'      * in other words, imagine a nested loop, from outer to inner:
#'         * for each cluster, for each category, for each outcome
#'   * numChoices: integer vector, number of outcomes for each category
#'   * classes: integer, number of classes (or clusters)
#' @return list of length n_category. For the ith entry, it contains a
#' matrix of outcome probabilities with dimensions n_class x n_outcomes[i]
poLCAParallel.unvectorize <- function(vp) {
    num_choices <- vp$numChoices
    n_category <- length(num_choices)
    # allocate matrices to the return list
    probs <- list()
    for (j in seq_len(n_category)) {
        probs[[j]] <- matrix(nrow = vp$classes, ncol = num_choices[j])
    }
    # copy over probabilities
    index <- 1
    for (m in 1:vp$classes) {
        for (j in seq_len(n_category)) {
            next_index <- index + num_choices[j] - 1
            probs[[j]][m, ] <- vp$vecprobs[index:next_index]
            index <- next_index + 1
        }
    }
    return(probs)
}
