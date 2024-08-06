# Example script comparing the standard errors with and without smoothing
# Clustering

nrep <- 32
nclass <- 5

data(carcinoma, package = "poLCAParallel")
dat <- carcinoma
f <- cbind(A, B, C, D, E, F, G) ~ 1

# fit poLCA without smoothing
set.seed(999204567)
lca <- poLCAParallel::poLCA(
  f, dat,
  nclass = nclass, nrep = nrep,
  verbose = FALSE, se.smooth = FALSE
)

# fit poLCA with smoothing
set.seed(999204567)
lca_smooth <- poLCAParallel::poLCA(
  f, dat,
  nclass = nclass, nrep = nrep,
  verbose = FALSE, se.smooth = TRUE
)

cat("Compare P.se\n")
cat("=====\n")
cat("No smoothing\n")
print(lca$P.se)
cat("With smoothing\n")
print(lca_smooth$P.se)

cat("\n")
cat("Compare probs.se\n")
cat("=====\n")
cat("No smoothing\n")
print(lca$probs.se)
cat("With smoothing\n")
print(lca_smooth$probs.se)

# Without smoothing, standard errors for class and response probabilities can be
# extremely small, eg 0.000000e+00 and 2.668947e-32. This usually happens when
# the fitted probabilites are very close to zero or one.
#
# With smoothing, we add a bit of bias to the class and response probabilities
# so that the reported standard errors are a bit larger. This is useful to
# to avoid reporting standard errors of 0.000000e+00.
