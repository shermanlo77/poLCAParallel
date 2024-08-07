# Example script comparing the standard errors with and without smoothing
# Clustering with regression

nrep <- 10000
nclass <- 5

data(cheating, package = "poLCAParallel")
dat <- cheating
f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA

# fit poLCA without smoothing
set.seed(999204567)
lca <- poLCAParallel::poLCA(
  f, dat,
  nclass = nclass, nrep = nrep, n.thread = 8,
  verbose = FALSE, se.smooth = FALSE
)

# fit poLCA with smoothing
set.seed(999204567)
lca_smooth <- poLCAParallel::poLCA(
  f, dat,
  nclass = nclass, nrep = nrep, n.thread = 8,
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
print(lca$probs)
print(lca$probs.se)
cat("With smoothing\n")
print(lca_smooth$probs.se)

# Without smoothing, standard errors for class and response probabilities can be
# extremely small, eg 0.00000000 and 2.509529e-81. This usually happens when
# the fitted probabilites are very close to zero or one.
#
# With smoothing, we add a bit of bias to the class and response probabilities
# so that the reported standard errors are a bit larger. This is useful to
# to avoid reporting standard errors of 0.00000000.
#
# However, this is still an experimental feature. For example, you can get
# a standard error greater than 1.0, eg 1.81408937, which is very large
# considering probabilities take values between 0.0 and 1.0.
