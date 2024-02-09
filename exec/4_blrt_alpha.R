# Example script on doing BLRT not using poLCAParallel::blrt()
#
# Reimplements 3_blrt.R but without using poLCAParallel::blrt(). Thus this
# script and 3_blrt.R should produce similar results as a verification.
#
# Even though this is a multi-core process, this is slower than 3_blrt.R

nrep <- 32 # number of different initial values
n_class_max <- 10 # maximum number of classes to investigate
n_bootstrap <- 100 # number of bootstrap samples
set.seed(1746091653)

# carcinoma is the sample data from poLCA
data(carcinoma, package = "poLCAParallel")
data_og <- carcinoma
data_column_names <- colnames(data_og)
formula <- cbind(A, B, C, D, E, F, G) ~ 1

# fit the model onto the data for different number of classes
# save the fitted model into model_array
model_array <- list()
for (nclass in 1:n_class_max) {
  model <- poLCAParallel::poLCA(
    formula, data_og,
    nclass = nclass, nrep = nrep, verbose = FALSE
  )
  model_array[[nclass]] <- model
}

# store p values for each nclass, 1 to n_class_max
# store 0 for 1 number of class, ie this says you cannot have zero number of
# classes
p_value_array <- c(0)
# for all number of classes investigated:
#   - store the log likelihood ratio
#   - store all bootstrap samples log likelihoods ratios
fitted_log_ratio_array <- rep(NaN, n_class_max)
bootstrap_log_ratio_array <- list()

# do the bootstrap likelihood ratio test for each number of classes
for (nclass in 2:n_class_max) {

  # get the null and alt models
  # these are models with one number of class differences
  null_model <- model_array[[nclass - 1]]
  alt_model <- model_array[[nclass]]

  # for each bootstrap sample, store the log likelihood ratio here
  bootstrap_log_likelihood_ratio <- c()

  # for each bootstrap sample
  for (b in seq_len(n_bootstrap)) {
    # get a bootstrap sample

    # make a copy from the original data
    # because polca will automatically remove columns containing all the same
    # values
    # the bootstrap samples should preserve the columns name for consistency
    data_bootstrap <- data_og

    # do a parametric bootstrap
    # data_bootstrap_some_columns may be missing columns as explained above
    data_bootstrap_some_columns <- poLCAParallel::poLCA.simdata(
      null_model$Nobs, null_model$probs, P=null_model$P
    )$dat
    colnames(data_bootstrap_some_columns) <- colnames(null_model$y)

    # replace the original data with bootstrap values
    data_bootstrap[colnames(null_model$y)] <- data_bootstrap_some_columns

    # fit the null model onto the bootstrap sample
    null_model_bootstrap <- poLCAParallel::poLCA(
      formula, data_bootstrap,
      nclass = nclass - 1, nrep = nrep, verbose = FALSE
    )

    # fit the alt model onto the bootstrap sample
    alt_model_bootstrap <- poLCAParallel::poLCA(
      formula, data_bootstrap,
      nclass = nclass, nrep = nrep, verbose = FALSE
    )

    # record the log likelihood ratio when using this bootstrap sample
    # this compares the fitted null and alt model
    bootstrap_log_likelihood_ratio <- c(
      bootstrap_log_likelihood_ratio,
      2 * alt_model_bootstrap$llik - 2 * null_model_bootstrap$llik
    )

  }

  # log likelihood ratio to compare the two models
  log_likelihood_ratio <- 2 * alt_model$llik - 2 * null_model$llik
  fitted_log_ratio_array[nclass] <- log_likelihood_ratio
  # store the log likelihoods ratios for all bootstrap samples
  bootstrap_log_ratio_array[[nclass]] <- bootstrap_log_likelihood_ratio
  # calculate the p value using all bootstrap values for this nclass
  p <- sum(bootstrap_log_likelihood_ratio > log_likelihood_ratio) / n_bootstrap
  # store the p value for this nclass
  p_value_array <- c(p_value_array, p)

}

# plot the bootstrap distribution of the log likelihood ratios for each class
# the red line shows the log likelihood ratio using the real data
pdf("4_blrt_alpha_llik.pdf")
boxplot(bootstrap_log_ratio_array,
  xlab = "number of classses", ylab = "log likelihood ratio"
)
# also plot the log likelihood ratio when using the real data
lines(1:n_class_max, fitted_log_ratio_array,
  type = "b", col = "red", pch = 15
)

# Plot the p value for each number of class.
# The p value is the proportion of bootstrap samples with log likelihood ratios
# greater than using the real data.
# Looking at the plot, I would select 3 or 4 classes as this is the biggest
# class with a small p value.
# Additional notes on how to interpret this graph:
#   - a low p value for a number of classes k suggest k number of classes is
#     better than k-1, so you would expect to see low p values for low number of
#     classes until you reach the optimal number of classes
#   - when the data follows the null hypothesis, the p value would follow a
#     uniform distribution, so for a class number too high, it should fluctuate
#     randomly between 0 and 1
# the solid line is at 5%
pdf("4_blrt_alpha_p_values.pdf")
barplot(p_value_array,
  xlab = "number of classes", ylab = "p-value",
  names.arg = 1:n_class_max
)
abline(h = 0.05)
