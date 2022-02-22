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
  results$Gsq <- goodness_fit_results[[2]]
  results$Chisq <- goodness_fit_results[[3]]
  return(results)
}