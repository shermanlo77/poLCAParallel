poLCAParallel.unvectorize = function(vp) {
  numChoices = vp$numChoices
  nCategory = length(numChoices)
  probs = list()
  for (j in 1:nCategory) {
    probs[[j]] = matrix(nrow=vp$classes, ncol=numChoices[j])
  }
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