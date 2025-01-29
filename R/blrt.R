#' Bootstrap likelihood ratio test (no regression only)
#'
#' Does the bootstrap likelihood ratio test. Provide two fitted models, the
#' null model and the alt model which fits a different number of clusters.
#' Bootstrap samples are generated using the null model. The null model and alt
#' model are refitted on the bootstrap samples to investigate the log likelihood
#' ratio of the two models.
#'
#' Runs in parallel for each bootstrap sample, potentially high memory if the
#' data is large
#'
#' @param model_null Fitted poLCA object, the null model
#' @param model_alt Fitted poLCA object, the alt model
#' @param n_bootstrap Number of bootstrap samples
#' @param n_thread Number of threads
#' @param n_rep Number of initial values to try when fitting on the bootstrap
#' samples
#' @param max_iter Maximum number of iterations for EM algorithm
#' @param tol Tolerance for difference in log likelihood, used for
#' stopping condition
#'
#' @return List containing the following:
#' <ul>
#'   <li>fitted_log_ratio: log likelihood ratio comparing the null and alt
#' model</li>
#'   <li>bootstrap_log_ratio: vector of length n_bootstrap, bootstrapped log
#' likelihood ratio comparing the null and alt model</li>
#'   <li>p_value<: the porportion of bootstrap samples with log likelihood
#' ratios greater than the fitted log likelihood ratio/li>
#' </ul>
#' @export
blrt <- function(model_null, model_alt, n_bootstrap,
                 n_thread = parallel::detectCores(), n_rep = 1, max_iter = 1000,
                 tol = 1e-10) {

  # extract fitted variables from the null model
  prior_null <- model_null$P
  prob_null <- poLCAParallel::poLCAParallel.vectorize(model_null$probs)$vecprobs
  n_cluster_null <- length(prior_null)

  # extract fitted variables from the alt model
  prior_alt <- model_alt$P
  prob_alt <- poLCAParallel::poLCAParallel.vectorize(model_alt$probs)$vecprobs
  n_cluster_alt <- length(prior_alt)

  # extract other information, use the null model
  n_data <- model_null$N
  n_outcomes <- apply(model_null$y, 2, max)

  # random seed required to generate new initial values when needed
  seed <- sample.int(
    as.integer(.Machine$integer.max), 5,
    replace = TRUE
  )

  bootstrap_log_ratio_array <- BlrtRcpp(
    prior_null, prob_null, n_cluster_null, prior_alt,
    prob_alt, n_cluster_alt, n_data, n_outcomes, n_bootstrap, n_rep,
    n_thread, max_iter, tol, seed
  )

  fitted_log_ratio <- 2 * model_alt$llik - 2 * model_null$llik

  p_value <- sum(bootstrap_log_ratio_array > fitted_log_ratio) / n_bootstrap

  to_return <- list()
  to_return[["fitted_log_ratio"]] <- fitted_log_ratio
  to_return[["bootstrap_log_ratio"]] <- bootstrap_log_ratio_array
  to_return[["p_value"]] <- p_value
  return(to_return)
}
