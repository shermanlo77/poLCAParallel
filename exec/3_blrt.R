# example script on bootstrap likelihood ratio test using poLCA and their sample
# data

n_thread <- 32
nrep <- 32 # number of different initial values (could be n_thread too)
n_class_max <- 10 # maximum number of classes to investigate
n_bootstrap <- 100 # number of bootstrap samples

# carcinoma is the sample data from poLCA
data(carcinoma, package = "poLCAParallel")
data_og <- carcinoma
data_column_names <- colnames(data_og)
f <- cbind(A, B, C, D, E, F, G) ~ 1

# fit the model onto the data for different number of classes
# save the fitted model into model_array
# you have already have done this before
model_array <- list()
for (nclass in 1:n_class_max) {
  model <- poLCAParallel::poLCA(
    f, data_og,
    nclass = nclass, nrep = nrep, n.thread = n_thread,
    verbose = FALSE
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
log_likelihood_ratio_array <- rep(NaN, n_class_max)
bootstrap_log_likelihood_ratio_array <- list()

# do the bootstrap likelihood ratio test for each number of classes
for (nclass in 2:n_class_max) {

  # get the null and alt models
  # these are models with one number of class differences
  null_model <- model_array[[nclass - 1]]
  alt_model <- model_array[[nclass]]

  # log likelihood ratio to compare the two models
  log_likelihood_ratio <- 2 * alt_model$llik - 2 * null_model$llik
  log_likelihood_ratio_array[nclass] <- log_likelihood_ratio

  # for each bootstrap sample, store the log likelihood ratio here
  bootstrap_log_likelihood_ratio <- poLCAParallel::blrt(
    null_model, alt_model,
    n_bootstrap, n_thread, nrep
  )

  # store the log likelihoods ratios for all bootstrap samples
  bootstrap_log_likelihood_ratio_array[[nclass]] <-
    bootstrap_log_likelihood_ratio
  # calculate the p value using all bootstrap values for this nclass
  p <- sum(bootstrap_log_likelihood_ratio > log_likelihood_ratio) / n_bootstrap
  p_value_array <- c(p_value_array, p)
}

# plot the p value for each number of class
# looking at the plot, I would select 3 classes as this is the biggest class
# with a small p value.
# Additional notes on how to interpret this graph:
#   - a low p value for a number of classes k suggest k number of classes is
#     better than k-1, so you would expect to see low p values for low number of
#     classes until you reach the optimal number of classes
#   - when the data follows the null hypothesis, the p value would follow a
#     uniform distribution, so for a class number too high, it should fluctuate
#     randomly between 0 and 1
# the solid line is at 5%
barplot(p_value_array,
  xlab = "number of classes", ylab = "p-value",
  names.arg = 1:n_class_max
)
abline(h = 0.05)


# plot the bootstrap distribution of the log likelihood ratios for each class
boxplot(bootstrap_log_likelihood_ratio_array,
  xlab = "number of classses", ylab = "log likelihood ratio"
)
# also plot the log likelihood ratio when using the real data
lines(1:n_class_max, log_likelihood_ratio_array,
  type = "b", col = "red", pch = 15
)
