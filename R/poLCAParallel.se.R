
#' Calculate the standard errors and adds them to the poLCA object
#'
#' R wrapper function for the C++ function StandardError Rcpp
#'
#' Calculate the standard errors and adds them, as attributes, to the poLCA
#' object, $P.se, $probs.se $coeff.se and $coeff.V
#'
#' $coeff.se and $coeff.V are set to NA if the provided poLCA is a standard
#' poLCA with no regression
#'
#' @param polca the resulting poLCA object from calling poLCA()
#' @param is_smooth Logical, experimental, for calculating the standard errors,
#'     whether to smooth the outcome probabilities to produce more numerical
#'     stable results as a cost of bias.
#'
#' @return the poLCA object with the attributes $P.se, $probs.se, $coeff.se and
#' $coeff.V modified or added
#'
#' @export
poLCAParallel.se <- function(polca, is_smooth = FALSE) {
  # extract required variables (or attributes)
  y <- polca$y
  formula <- formula(
    paste0("cbind(", paste(colnames(y), collapse = ","), ")~1")
  )
  mframe <- model.frame(formula, y, na.action = NULL)
  responses <- model.response(mframe)
  responses[is.na(responses)] <- 0

  features <- as.matrix(polca$x)
  n_data <- nrow(features)
  n_feature <- ncol(features)
  prob_vec <- poLCAParallel.vectorize(polca$probs)
  probs <- prob_vec$vecprobs
  n_outcomes <- prob_vec$numChoices
  n_cluster <- prob_vec$classes
  prior <- polca$prior
  posterior <- polca$posterior

  # call the C++ function
  results <- StandardErrorRcpp(
    features, responses, probs, prior, posterior,
    n_data, n_feature, n_outcomes, n_cluster, is_smooth
  )

  # standard errors for the prior
  polca$P.se <- results[[1]]

  # standard errors for the outcome probabilities
  prob_vec$vecprobs <- results[[2]]
  polca$probs.se <- poLCAParallel.unvectorize(prob_vec)
  names(polca$probs.se) <- colnames(y)

  # standard errors for the coefficients
  if (n_feature > 1) {
    polca$coeff.V <- results[[3]]
    coeff_se <- matrix(sqrt(diag(polca$coeff.V)),
      nrow = n_feature,
      ncol = (n_cluster - 1)
    )
    rownames(coeff_se) <- rownames(polca$coeff)
    polca$coeff.se <- coeff_se
  } else {
    polca$coeff.se <- NA
    polca$coeff.V <- NA
  }

  return(polca)
}
