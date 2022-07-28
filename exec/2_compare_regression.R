# Comparing poLCA with poLCAParallel for clustering with regression
#
# Fits poLCA and poLCAParallel for different number of classes/clusters and for
# different sample datasets. Does benchmark and compares how similar each
# property of the results are
#
# Requires poLCA and poLCAParallel
#
# poLCA requires attaching MASS for standard errors
#
# Note: Not sure how to compare coefficients, the clusters/classes are not
# ordered and the dimension of the coefficient is #clusters - 1

library(MASS)
nrep <- 32
n_thread <- 1

# for high number of classes, you get into numerical errors, this is where
# poLCA and poLCAParallel diverge in methodology.
#
# arma warnings are supressed using #define ARMA_WARN_LEVEL 1
# remove it to see supressed warning messages
for (nclass in 2:5) {
  for (i in 1:2) {
    if (i == 1) {
      data(cheating)
      dat <- cheating
      f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA
      cat("========== cheating ==========")
    } else {
      data(election)
      dat <- election
      f <- cbind(
        MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG,
        MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB
      ) ~ PARTY
      cat("========== election ==========")
    }
    cat("\n")
    cat(paste("==========", nclass, "classes ==========\n"))

    # using original code
    set.seed(0)
    start_time <- Sys.time()
    lca <- poLCA::poLCA(
      f, dat,
      nclass = nclass, nrep = nrep, verbose = FALSE
    )
    diff_time_og <- Sys.time() - start_time
    units(diff_time_og) <- "secs"

    # using parallel code
    set.seed(0)
    start_time <- Sys.time()
    lca_parallel <- poLCAParallel::poLCA(
      f, dat,
      nclass = nclass, nrep = nrep, n.thread = n_thread,
      verbose = FALSE
    )
    diff_time_parallel <- Sys.time() - start_time
    units(diff_time_parallel) <- "secs"

    # compare timings
    cat(paste("Time for orginal code", diff_time_og, "s\n"))
    cat(paste("Time for parallel code", diff_time_parallel, "s\n"))

    # compare results
    cat("\n")
    cat("Compare results, TRUE if exactly the same\n")
    cat(paste("llik:", all.equal(lca$llik, lca_parallel$llik), "\n"))
    cat(
      paste("attempts:", all.equal(lca$attempts, lca_parallel$attempts), "\n")
    )
    cat(paste("aic:", all.equal(lca$aic, lca_parallel$aic), "\n"))
    cat(paste("bic:", all.equal(lca$bic, lca_parallel$bic), "\n"))
    cat(paste("Nobs:", all.equal(lca$Nobs, lca_parallel$Nobs), "\n"))
    cat(paste("Chisq:", all.equal(lca$Chisq, lca_parallel$Chisq), "\n"))
    # note: predcell in original code rounds
    cat(
      paste("predcell:", all.equal(lca$predcell, lca_parallel$predcell), "\n")
    )
    cat(paste("Gsq:", all.equal(lca$Gsq, lca_parallel$Gsq), "\n"))
    cat(paste("y:", all.equal(lca$y, lca_parallel$y), "\n"))
    cat(paste("x:", all.equal(lca$x, lca_parallel$x), "\n"))

    # compare results, but classes/clusters do need to be sorted
    index <- order(lca$P)
    index_parallel <- order(lca_parallel$P)

    cat(paste(
      "P:",
      all.equal(lca$P[index], lca_parallel$P[index_parallel]), "\n"
    ))

    cat(paste(
      "P.se:",
      all.equal(lca$P.se[index], lca_parallel$P.se[index_parallel]), "\n"
    ))

    cat(paste("posterior:", all.equal(
      lca$posterior[, index], lca_parallel$posterior[, index_parallel]
    ), "\n"))

    probs <- poLCA::poLCA.reorder(lca$probs, index)
    probs_parallel <- poLCA::poLCA.reorder(lca_parallel$probs, index_parallel)
    cat(paste("probs:", all.equal(probs, probs_parallel), "\n"))

    probs_se <- poLCA::poLCA.reorder(lca$probs.se, index)
    probs_se_parallel <- poLCA::poLCA.reorder(
      lca_parallel$probs.se, index_parallel
    )
    cat(paste("probs.se:", all.equal(probs_se, probs_se_parallel), "\n"))

    cat("\n")
  }
}
