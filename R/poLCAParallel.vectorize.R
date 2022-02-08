poLCAParallel.vectorize = function(probs) {
  classes = nrow(probs[[1]])
  vecprobs = c()
  for (m in 1:classes) {
    for (j in 1:length(probs)) {
      vecprobs = c(vecprobs, probs[[j]][m, ])
    }
  }
  numChoices = sapply(probs,ncol)
  return(list(vecprobs=vecprobs, numChoices=numChoices, classes=classes))
}
