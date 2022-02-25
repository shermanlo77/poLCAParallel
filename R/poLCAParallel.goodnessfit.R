#' poLCAParallel - Goodness of Fit
#' Add predcell, Gsq and Chisq to a fitted poLCA object (or a list)
#'
#' Find unique responses and put them in a dataframe along with the observed and
    #' expected frequencies. Also calculates the log likelihood ratio and chi
    #' squared statistics
#' @param results a poLCA object (see poLCA.R)
#' @return list with 3 items:
    #' predcell: dataframe of unique responses with their observed and expected
        #' frequencies
    #' Gsq: log likelihood ratio
    #' Chisq: chi squared statistic
poLCAParallel.goodnessfit = function(results) {
    y = results$y
    prob_vec = poLCAParallel.vectorize(results$probs);
    goodness_fit_results = GoodnessFitRcpp(t(y),
                                           results$P,
                                           prob_vec$vecprobs,
                                           results$N,
                                           length(prob_vec$numChoices),
                                           prob_vec$numChoices,
                                           prob_vec$classes)
    results$predcell = data.frame(goodness_fit_results[[1]])
    colnames(results$predcell) = c(colnames(y), "observed", "expected")
    results$Gsq = goodness_fit_results[[2]]
    results$Chisq = goodness_fit_results[[3]]
    return(results)
}
