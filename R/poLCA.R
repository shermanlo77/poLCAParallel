#' Estimates latent class and latent class regression models for polytomous
#' outcome variables
#'
#' Latent class analysis, also known as latent structure analysis, is a
#' technique for the analysis of clustering among observations in multi-way
#' tables of qualitative/categorical variables.  The central idea is to fit a
#' model in which any confounding between the manifest variables can be
#' explained by a single unobserved "latent" categorical variable. `poLCA` uses
#' the assumption of local independence to estimate a mixture model of latent
#' multi-way tables, the number of which (`nclass`) is specified by the user.
#' Estimated parameters include the class-conditional response probabilities for
#' each manifest variable, the "mixing" proportions denoting population share of
#' observations corresponding to each latent multi-way table, and coefficients
#' on any class-predictor covariates, if specified in the model.
#'
#' Model specification: Latent class models have more than one manifest
#' variable, so the response variables are `cbind(dv1,dv2,dv3...)` where
#' `dv#` refer to variable names in the data frame.  For models with no
#' covariates, the formula is `cbind(dv1,dv2,dv3)~1`.  For models with
#' covariates, replace the `~1` with the desired function of predictors
#' `iv1,iv2,iv3...` as, for example, `cbind(dv1,dv2,dv3)~iv1+iv2*iv3`.
#'
#' `poLCA` treats all manifest variables as qualitative/categorical/nominal
#' -- NOT as ordinal.
#'
#' The library poLCAParallel reimplements poLCA in C++. This was done using
#' [Rcpp](https://cran.r-project.org/web/packages/Rcpp) and
#' [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo) which
#' allows C++ code to interact with R. Additional notes include:
#'
#' * The code uses [Armadillo](http://arma.sourceforge.net/) for linear algebra
#' * Multiple repetitions are done in parallel using
#'   [`<thread>`](https://www.cplusplus.com/reference/thread/) for multi-thread
#'   programming and [`<mutex>`](https://www.cplusplus.com/reference/mutex/) to
#'   prevent data races
#' * Response probabilities are reordered to increase cache efficiency
#' * Use of [`std::map`](https://en.cppreference.com/w/cpp/container/map) for
#'   the chi-squared calculations
#'
#' Further reading available on a
#' [QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages.html).
#'
#' References:
#' * Agresti, Alan. 2002. *Categorical Data Analysis, second edition*.
#'   Hoboken: John Wiley \& Sons.
#' * Bandeen-Roche, Karen, Diana L. Miglioretti, Scott L. Zeger, and Paul J.
#'   Rathouz. 1997. "Latent Variable Regression for Multiple Discrete
#'   Outcomes." *Journal of the American Statistical Association*.
#'   92(440): 1375-1386.
#' * Hagenaars, Jacques A. and Allan L. McCutcheon, eds. 2002.
#'   *Applied Latent Class Analysis*. Cambridge: Cambridge University
#'   Press.
#' * McLachlan, Geoffrey J. and Thriyambakam Krishnan. 1997.
#'   *The EM Algorithm and Extensions*. New York: John Wiley \& Sons.
#'
#' Notes:
#'
#' `poLCA` uses EM and Newton-Raphson algorithms to maximize the latent
#' class model log-likelihood function. Depending on the starting parameters,
#' this algorithm may only locate a local, rather than global, maximum. This
#' becomes more and more of a problem as `nclass` increases. It is
#' therefore highly advisable to run `poLCA` multiple times until you are
#' relatively certain that you have located the global maximum log-likelihood.
#' As long as `probs.start=NULL`, each function call will use different
#' (random) initial starting parameters.  Alternatively, setting `nrep` to
#' a value greater than one enables the user to estimate the latent class model
#' multiple times with a single call to `poLCA`, thus conducting the
#' search for the global maximizer automatically.
#'
#' The term "Latent class regression" (LCR) can have two meanings.  In this
#' package, LCR models refer to latent class models in which the probability of
#' class membership is predicted by one or more covariates.  However, in other
#' contexts, LCR is also used to refer to regression models in which the
#' manifest variable is partitioned into some specified number of latent classes
#' as part of estimating the regression model. It is a way to simultaneously fit
#' more than one regression to the data when the latent data partition is
#' unknown. The `flexmix`` function in package `flexmix`` will estimate this
#' other type of LCR model.  Because of these terminology issues, the LCR models
#' this package estimates are sometimes termed "latent class models with
#' covariates" or "concomitant-variable latent class analysis," both of which
#' are accurate descriptions of this model.
#'
#' @param formula A formula expression of the form `response ~ predictors`.
#'     The details of model specification are given below.
#' @param data A data frame containing variables in `formula`. Manifest
#'     variables must contain *only* integer values, and must be coded with
#'     consecutive values from 1 to the maximum number of outcomes for each
#'     variable. All missing values should be entered as `NA`.
#' @param nclass The number of latent classes to assume in the model. Setting
#'     `nclass=1` results in `poLCA` estimating the loglinear
#'     independence model. The default is two.
#' @param maxiter The maximum number of iterations through which the estimation
#'     algorithm will cycle.
#' @param graphs Logical, for whether `poLCA` should graphically display
#'     the parameter estimates at the completion of the estimation algorithm.
#'     The default is `FALSE`.
#' @param tol A tolerance value for judging when convergence has been reached.
#'     When the one-iteration change in the estimated log-likelihood is less
#'     than `tol`, the estimation algorithm stops updating and considers
#'     the maximum log-lparameterikelihood to have been found.
#' @param na.rm Logical, for how `poLCA` handles cases with missing values
#'     on the manifest variables. If `TRUE`, those cases are removed
#'     (listwise deleted) before estimating the model. If `FALSE`, cases
#'     with missing values are retained. Cases with missing covariates are
#'     always removed. The default is `TRUE`.
#' @param probs.start A list of matrices of class-conditional response
#'     probabilities to be used as the starting values for the estimation
#'     algorithm. Each matrix in the list corresponds to one manifest variable,
#'     with one row for each latent class, and one column for each outcome. The
#'     default is `NULL`, producing random starting values.  Note that if
#'     `nrep>1`, then any user-specified `probs.start` values are only
#'     used in the first of the `nrep` attempts.
#' @param nrep Number of times to estimate the model, using different values of
#'     `probs.start`. The default is one.  Setting `nrep`>1 automates
#'     the search for the global---rather than just a local---maximum of the
#'     log-likelihood function. `poLCA` returns the parameter estimates
#'     corresponding to the model with the greatest log-likelihood.
#' @param verbose Logical, indicating whether `poLCA` should output to the
#'     screen the results of the model. If `FALSE`, no output is produced.
#'     The default is `TRUE`.
#' @param cal.se Logical, indicating whether `poLCA` should calculate the
#'     standard errors of the estimated class-conditional response probabilities
#'     and mixing proportions. default is `TRUE`.
#' @param calc.chisq Logical, indicate whether to calculate the goodness of fit
#'     statistics, the chi squared statistics and the log likelihood ratio.
#'     The default is `TRUE`.
#' @param n.thread Integer, the number of threads used to run each repetition.
#'
#' @return an object of class poLCA; a list containing the following elements:
#'   * y: data frame of manifest variables.
#'   * x: data frame of covariates, if specified.
#'   * N: number of cases used in model.
#'   * Nobs: number of fully observed cases (less than or equal to `N`).
#'   * probs: estimated class-conditional response probabilities.
#'   * probs.se: standard errors of estimated class-conditional response
#'     probabilities, in the same format as `probs`.
#'   * P: sizes of each latent class; equal to the mixing proportions in the
#'     function basic latent class model, or the mean of the priors in the
#'     latent class regression model.
#'   * P.se: the standard errors of the estimated `P`.
#'   * posterior: matrix of posterior class membership probabilities; also see
#'     function 'poLCA.posterior'.
#'   * predclass: vector of predicted class memberships, by modal assignment.
#'   * predcell: table of observed versus predicted cell counts for cases with
#'     no missing values; also see functions `poLCA.table` and `poLCA.predcell`
#'   * llik: maximum value of the log-likelihood.
#'   * numiter: number of iterations until reaching convergence.
#'   * maxiter: maximum number of iterations through which the estimation
#'     algorithm was set to run.
#'   * coeff: multinomial logit coefficient estimates on covariates (when
#'     estimated). `coeff` is a matrix with `nclass-1` columns, and one row for
#'     each covariate.  All logit coefficients are calculated for classes with
#'     respect to class 1.
#'   * coeff.se: standard errors of coefficient estimates on covariates (when
#'     estimated), in the same format as `coeff`.
#'   * coeff.V: covariance matrix of coefficient estimates on covariates (when
#'     estimated).
#'   * aic: Akaike Information Criterion.
#'   * bic: Bayesian Information Criterion.
#'   * Gsq: Likelihood ratio/deviance statistic.
#'   * Chisq: Pearson Chi-square goodness of fit statistic for fitted vs.
#'     observed multiway tables.
#'   * time: length of time it took to run the model.
#'   * npar: number of degrees of freedom used by the model (estimated
#'     parameters).
#'   * resid.df: number of residual degrees of freedom.
#'   * attempts: a vector containing the maximum log-likelihood values found in
#'     each of the `nrep` attempts to fit the model.
#'   * eflag: Logical, error flag. `TRUE` if estimation algorithm needed to
#'     automatically restart with new initial parameters. A restart is caused in
#'     the event of computational/rounding errors that result in nonsensical
#'     parameter estimates.
#'   * probs.start: A list of matrices containing the class-conditional response
#'     probabilities used as starting values in the estimation algorithm. If the
#'     algorithm needed to restart (see `eflag`), then this contains the
#'     starting values used for the final, successful, run.
#'   * probs.start.ok: Logical. `FALSE` if `probs.start` was incorrectly
#'     specified by the user, otherwise `TRUE`.
#'   * call: function call to `poLCA`.
#'
#' @example
#' ##
#' ## Three models without covariates:
#' ## M0: Loglinear independence model.
#' ## M1: Two-class latent class model.
#' ## M2: Three-class latent class model.
#' ##
#' data(values)
#' f <- cbind(A,B,C,D)~1
#' M0 <- poLCA(f,values,nclass=1) # log-likelihood: -543.6498
#' M1 <- poLCA(f,values,nclass=2) # log-likelihood: -504.4677
#' M2 <- poLCA(f,values,nclass=3,maxiter=8000) # log-likelihood: -503.3011
#'
#' ##
#' ## Three-class model with a single covariate.
#' ##
#' data(election)
#' f2a <- cbind(MORALG,CARESG,KNOWG,LEADG,DISHONG,INTELG,
#'              MORALB,CARESB,KNOWB,LEADB,DISHONB,INTELB)~PARTY
#' nes2a <- poLCA(f2a,election,nclass=3,nrep=5)    # log-likelihood: -16222.32
#' pidmat <- cbind(1,c(1:7))
#' exb <- exp(pidmat \%*\% nes2a$coeff)
#' matplot(c(1:7),(cbind(1,exb)/(1+rowSums(exb))),ylim=c(0,1),type="l",
#'     main="Party ID as a predictor of candidate affinity class",
#'     xlab="Party ID: strong Democratic (1) to strong Republican (7)",
#'     ylab="Probability of latent class membership",lwd=2,col=1)
#' text(5.9,0.35,"Other")
#' text(5.4,0.7,"Bush affinity")
#' text(1.8,0.6,"Gore affinity")
#'
#' @export
poLCA <- function(formula,
                  data,
                  nclass = 2,
                  maxiter = 1000,
                  graphs = FALSE,
                  tol = 1e-10,
                  na.rm = TRUE,
                  probs.start = NULL,
                  nrep = 1,
                  verbose = TRUE,
                  calc.se = TRUE,
                  calc.chisq = TRUE,
                  n.thread = parallel::detectCores()) {
    starttime <- Sys.time()
    mframe <- model.frame(formula, data, na.action = NULL)
    mf <- model.response(mframe)
    if (any(mf < 1, na.rm = TRUE) | any(round(mf) != mf, na.rm = TRUE)) {
        stop("\n ALERT: some manifest variables contain values that are not
              positive integers. For poLCA to run, please recode categorical
              outcome variables to increment from 1 to the maximum number of
              outcome categories for each variable. \n\n")
    }
    data <- data[rowSums(is.na(model.matrix(formula, mframe))) == 0, ]
    if (na.rm) {
        mframe <- model.frame(formula, data)
        y <- model.response(mframe)
    } else {
        mframe <- model.frame(formula, data, na.action = NULL)
        y <- model.response(mframe)
        y[is.na(y)] <- 0
    }
    if (any(sapply(lapply(as.data.frame(y), table), length) == 1)) {
        y <- y[, !(sapply(apply(y, 2, table), length) == 1)]
        cat("\n ALERT: at least one manifest variable contained only one
             outcome category, and has been removed from the analysis. \n\n")
    }
    x <- model.matrix(formula, mframe) # features
    N <- nrow(y) # number of data points
    J <- ncol(y) # number of categories
    K.j <- t(matrix(apply(y, 2, max))) # number of outcomes for each category
    R <- nclass
    S <- ncol(x) # number of features
    eflag <- FALSE # set to TRUE if find an error
    probs.start.ok <- TRUE
    ret <- list() # list of items to return

    # nclass == 1 will use original code
    # poLCAParallel edits the orginial code for the nclass > 1 case
    if (R == 1) {
        ret$probs <- list()
        for (j in 1:J) {
            ret$probs[[j]] <- matrix(NA, nrow = 1, ncol = K.j[j])
            for (k in 1:K.j[j]) {
                ret$probs[[j]][k] <- sum(y[, j] == k) / sum(y[, j] > 0)
            }
        }
        ret$probs.start <- ret$probs
        ret$P <- 1
        prior <- matrix(1, nrow = N, ncol = 1)
        ret$predclass <- prior
        ret$posterior <- ret$predclass
        ret$llik <- sum(log(poLCA.ylik.C(poLCA.vectorize(ret$probs), y)))
        if (calc.se) {
            se <- poLCA.se(y, x, ret$probs, prior, ret$posterior)
            # standard errors of class-conditional response probabilities
            ret$probs.se <- se$probs
            # standard errors of class population shares
            ret$P.se <- se$P
        } else {
            ret$probs.se <- NA
            ret$P.se <- NA
        }
        ret$numiter <- 1
        ret$probs.start.ok <- TRUE
        ret$coeff <- NA
        ret$coeff.se <- NA
        ret$coeff.V <- NA
        ret$eflag <- FALSE
        if (S > 1) {
            cat("\n ALERT: covariates not allowed when nclass=1;
                 will be ignored. \n \n")
            S <- 1
        }
    } else {
        # error checking on user-inputted probs.start
        if (!is.null(probs.start)) {
            if ((length(probs.start) != J) | (!is.list(probs.start))) {
                probs.start.ok <- FALSE
            } else {
                if (sum(sapply(probs.start, dim)[1, ] == R) != J) {
                    probs.start.ok <- FALSE
                }
                if (sum(sapply(probs.start, dim)[2, ] == K.j) != J) {
                    probs.start.ok <- FALSE
                }
                if (sum(round(sapply(probs.start, rowSums), 4) == 1)
                != (R * J)) {
                    probs.start.ok <- FALSE
                }
            }
        }

        # perpare initial values
        initial_prob <- list()
        initial_prob_vector <- c()
        irep <- 1

        # see if can use user provided probs.start
        if (probs.start.ok & !is.null(probs.start)) {
            initial_prob[[1]] <- poLCAParallel.vectorize(probs.start)
            initial_prob_vector <- c(
                initial_prob_vector,
                initial_prob[[1]]$vecprobs
            )
            irep <- irep + 1
        }

        # then generate random probabilities
        if (nrep > 1 | irep == 1) {
            for (repl in irep:nrep) {
                probs <- list()
                for (j in 1:J) {
                    probs[[j]] <- matrix(
                        runif(R * K.j[j]),
                        nrow = R, ncol = K.j[j]
                    )
                    probs[[j]] <- probs[[j]] / rowSums(probs[[j]])
                }
                initial_prob[[repl]] <- poLCAParallel.vectorize(probs)
                initial_prob_vector <- c(
                    initial_prob_vector,
                    initial_prob[[repl]]$vecprobs
                )
            }
        }

        # random seed required to generate new initial values when needed
        seed <- sample.int(
            as.integer(.Machine$integer.max), 5,
            replace = TRUE
        )
        # run C++ code here, extract results
        em_results <- EmAlgorithmRcpp(
            x,
            t(y),
            initial_prob_vector,
            N,
            S,
            J,
            K.j,
            R,
            nrep,
            n.thread,
            maxiter,
            tol,
            seed
        )
        rgivy <- em_results[[1]]
        prior <- em_results[[2]]
        estimated_prob <- em_results[[3]]
        b <- em_results[[4]]
        ret$attempts <- em_results[[5]]
        best_rep_index <- em_results[[6]]
        numiter <- em_results[[7]]
        best_initial_prob <- em_results[[8]]
        eflag <- em_results[[9]]

        llik <- ret$attempts[best_rep_index]

        # copy initial_prob[[1]] as it can be used as a parameter for
        # poLCAParallel.unvectorize(vp)
        # replace $vecprobs with the estimate probabilities
        vp <- initial_prob[[1]]
        vp$vecprobs <- estimated_prob

        ret$probs.start <- initial_prob[[1]]
        ret$probs.start$vecprobs <- best_initial_prob

        # calculate standard error
        if (calc.se) {
            se <- poLCA.se(
                y, x, poLCAParallel.unvectorize(vp),
                prior, rgivy
            )
            rownames(se$b) <- colnames(x)
        } else {
            se <- list(
                probs = NA, P = NA, b = matrix(nrow = S, ncol = R - 1),
                var.b = NA
            )
        }

        # labelling b
        if (S > 1) {
            b <- matrix(b, nrow = S)
            rownames(b) <- colnames(x)
        } else {
            b <- NA
            se$b <- NA
            se$var.b <- NA
        }

        # maximum value of the log-likelihood
        ret$llik <- llik
        # starting values of class-conditional response probabilities
        ret$probs.start <- poLCAParallel.unvectorize(ret$probs.start)
        # estimated class-conditional response probabilities
        ret$probs <- poLCAParallel.unvectorize(vp)
        # standard errors of class-conditional response probabilities
        ret$probs.se <- se$probs
        # standard errors of class population shares
        ret$P.se <- se$P
        # NxR matrix of posterior class membership probabilities
        ret$posterior <- rgivy
        # Nx1 vector of predicted class memberships, by modal assignment
        ret$predclass <- apply(ret$posterior, 1, which.max)
        # estimated class population shares
        ret$P <- colMeans(ret$posterior)
        # number of iterations until reaching convergence
        ret$numiter <- numiter
        # if starting probs specified, logical indicating proper entry
        # format
        ret$probs.start.ok <- probs.start.ok

        ret$coeff <- b # coefficient estimates (when estimated)
        # standard errors of coefficient estimates (when estimated)
        ret$coeff.se <- se$b
        # covariance matrix of coefficient estimates (when estimated)
        ret$coeff.V <- se$var.b

        # error flag, true if estimation algorithm ever needed to restart
        # with new initial values
        ret$eflag <- eflag
    }
    names(ret$probs) <- colnames(y)
    if (calc.se) {
        names(ret$probs.se) <- colnames(y)
    }
    # number of degrees of freedom used by the model (number of estimated
    # parameters)
    ret$npar <- (R * sum(K.j - 1)) + (R - 1)
    if (S > 1) {
        ret$npar <- ret$npar + (S * (R - 1)) - (R - 1)
    }
    # Akaike Information Criterion
    ret$aic <- (-2 * ret$llik) + (2 * ret$npar)
    # Schwarz-Bayesian Information Criterion
    ret$bic <- (-2 * ret$llik) + (log(N) * ret$npar)
    # number of fully observed cases (if na.rm=F)
    ret$Nobs <- sum(rowSums(y == 0) == 0)

    y[y == 0] <- NA
    ret$y <- data.frame(y) # outcome variables
    ret$x <- data.frame(x) # covariates, if specified
    for (j in 1:J) {
        rownames(ret$probs[[j]]) <- paste("class ", 1:R, ": ", sep = "")
        if (is.factor(data[, match(colnames(y), colnames(data))[j]])) {
            lev <- levels(data[, match(colnames(y), colnames(data))[j]])
            colnames(ret$probs[[j]]) <- lev
            ret$y[, j] <- factor(ret$y[, j], labels = lev)
        } else {
            colnames(ret$probs[[j]]) <-
                paste("Pr(", 1:ncol(ret$probs[[j]]), ")", sep = "")
        }
    }
    ret$N <- N # number of observations

    # if no rows are fully observed or chi squared not requested
    if ((all(rowSums(y == 0) > 0)) | !calc.chisq) {
        ret$Chisq <- NA
        ret$Gsq <- NA
        ret$predcell <- NA
    } else {
        ret <- poLCAParallel.goodnessfit(ret)
    }

    ret$maxiter <- maxiter # maximum number of iterations specified by user
    # number of residual degrees of freedom
    ret$resid.df <- min(ret$N, (prod(K.j) - 1)) - ret$npar
    class(ret) <- "poLCA"
    if (graphs) {
        plot.poLCA(ret)
    }
    if (verbose) {
        print.poLCA(ret)
    }
    ret$time <- Sys.time() - starttime # how long it took to run the model
    ret$call <- match.call()
    return(ret)
}
