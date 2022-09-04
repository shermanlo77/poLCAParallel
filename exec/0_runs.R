# Runs poLCAParallel for clustering with no regression
#
# Fits poLCAParallel for different number of classes/clusters and for
# different sample datasets. Does benchmark.
#
# Do not use library() and attach packages, this it to test if poLCAParallel has
# successfully installed and using dependent packages
#
# Requires poLCAParallel to be installed

nrep <- 32
n.thread <- 1

for (nclass in 2:5) {
  for (i in 1:5) {
    if (i == 1) {
      data(carcinoma, package = "poLCAParallel")
      dat <- carcinoma
      f <- cbind(A, B, C, D, E, F, G) ~ 1
      cat("========== carcinoma ==========")
    } else if (i == 2) {
      data(cheating, package = "poLCAParallel")
      dat <- cheating
      f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1
      cat("========== cheating ==========")
    } else if (i == 3) {
      data(election, package = "poLCAParallel")
      dat <- election
      f <- cbind(
        MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG,
        MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB
      ) ~ 1
      cat("========== election ==========")
    } else if (i == 4) {
      data(gss82, package = "poLCAParallel")
      dat <- gss82
      f <- cbind(PURPOSE, ACCURACY, UNDERSTA, COOPERAT) ~ 1
      cat("========== gss82 ==========")
    } else {
      data(values, package = "poLCAParallel")
      dat <- values
      f <- cbind(A, B, C, D) ~ 1
      cat("========== values ==========")
    }
    cat("\n")
    cat(paste("==========", nclass, "classes ==========\n"))

    # using parallel code
    set.seed(0)
    start_time <- Sys.time()
    lca_parallel <- poLCAParallel::poLCA(
      f, dat,
      nclass = nclass, nrep = nrep, n.thread = n.thread,
      verbose = FALSE
    )
    diff_time_parallel <- Sys.time() - start_time
    units(diff_time_parallel) <- "secs"

    # compare timings
    cat(paste("Time for parallel code", diff_time_parallel, "s\n"))
  }
}
