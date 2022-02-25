#' poLCAParallel - Unvectorize
#' Mimics poLCA.unvectorize but with some of the dimensions swapped
#'
#' Reverse the operations of poLCAParallel.vectorize
#' @param vp list of three items
    #' vecprobs: vector of outcome probabilities, a flatten list of matrices
        #' dim 0: for each outcome
        #' dim 1: for each category
        #' dim 2: for each cluster
    #' numChoices: integer vector, number of outcomes for each category
    #' classes: integer, number of classes (or clusters)
#' @return list of length n_category, for the ith entry, it contains a
    #' matrix of outcome probabilities with dimensions n_class x n_outcomes[i]
poLCAParallel.unvectorize = function(vp) {
    numChoices = vp$numChoices
    nCategory = length(numChoices)
    # allocate matrices to the return list
    probs = list()
    for (j in 1:nCategory) {
        probs[[j]] = matrix(nrow=vp$classes, ncol=numChoices[j])
    }
    # copy over probabilities
    index = 1
    for (m in 1:vp$classes) {
        for (j in 1:nCategory) {
            nextIndex = index + numChoices[j] - 1
            probs[[j]][m, ] = vp$vecprobs[index:nextIndex]
            index = nextIndex + 1
        }
    }
    return(probs)
}
