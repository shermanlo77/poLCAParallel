#' Estimates latent class and latent class regression models for polytomous
#' outcome variables
#'
#' @param formula A formula expression of the form \code{response ~ predictors}.
#'     The details of model specification are given below.
#' @param data A data frame containing variables in \code{formula}. Manifest
#'     variables must contain \emph{only} integer values, and must be coded with
#'     consecutive values from 1 to the maximum number of outcomes for each
#'     variable. All missing values should be entered as \code{NA}.
#' @param nclass The number of latent classes to assume in the model. Setting
#'     \code{nclass=1} results in \code{poLCA} estimating the loglinear
#'     independence model. The default is two.
#' @param maxiter The maximum number of iterations through which the estimation
#'     algorithm will cycle.
#' @param graphs Logical, for whether \code{poLCA} should graphically display
#'     the parameter estimates at the completion of the estimation algorithm.
#'     The default is \code{FALSE}.
#' @param tol A tolerance value for judging when convergence has been reached.
#'     When the one-iteration change in the estimated log-likelihood is less
#'     than \code{tol}, the estimation algorithm stops updating and considers
#'     the maximum log-likelihood to have been found.
#' @param na.rm Logical, for how \code{poLCA} handles cases with missing values
#'     on the manifest variables. If \code{TRUE}, those cases are removed
#'     (listwise deleted) before estimating the model. If \code{FALSE}, cases
#'     with missing values are retained. Cases with missing covariates are
#'     always removed. The default is \code{TRUE}.
#' @param probs.start A list of matrices of class-conditional response
#'     probabilities to be used as the starting values for the estimation
#'     algorithm. Each matrix in the list corresponds to one manifest variable,
#'     with one row for each latent class, and one column for each outcome. The
#'     default is \code{NULL}, producing random starting values.  Note that if
#'     \code{nrep>1}, then any user-specified \code{probs.start} values are only
#'     used in the first of the \code{nrep} attempts.
#' @param nrep Number of times to estimate the model, using different values of
#'     \code{probs.start}. The default is one.  Setting \code{nrep}>1 automates
#'     the search for the global---rather than just a local---maximum of the
#'     log-likelihood function. \code{poLCA} returns the parameter estimates
#'     corresponding to the model with the greatest log-likelihood.
#' @param verbose Logical, indicating whether \code{poLCA} should output to the
#'     screen the results of the model. If \code{FALSE}, no output is produced.
#'     The default is \code{TRUE}.
#' @param cal.se Logical, indicating whether \code{poLCA} should calculate the
#'     standard errors of the estimated class-conditional response probabilities
#'     and mixing proportions. The default is \code{TRUE}.
#' @param calc.chisq Logical, indicate whether to calculate the goodness of fit
#'     statistics, the chi squared statistics and the log likelihood ratio.
#'     The default is \code{TRUE}.
#' @param n.thread Integer, the number of threads used to run each repetition.
poLCA <- function(formula,
                  data,
                  nclass=2,
                  maxiter=1000,
                  graphs=FALSE,
                  tol=1e-10,
                  na.rm=TRUE,
                  probs.start=NULL,
                  nrep=1,
                  verbose=TRUE,
                  calc.se=TRUE,
                  calc.chisq=TRUE,
                  n.thread=parallel::detectCores()) {
    cat("\nUsing parallel version of poLCA\n")
    starttime <- Sys.time()
    mframe <- model.frame(formula, data, na.action=NULL)
    mf <- model.response(mframe)
    if (any(mf<1, na.rm=TRUE) | any(round(mf) != mf, na.rm=TRUE)) {
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
        mframe <- model.frame(formula, data, na.action=NULL)
        y <- model.response(mframe)
        y[is.na(y)] <- 0
    }
    if (any(sapply(lapply(as.data.frame(y), table), length) == 1)) {
        y <- y[, !(sapply(apply(y, 2, table), length) == 1)]
        cat("\n ALERT: at least one manifest variable contained only one
             outcome category, and has been removed from the analysis. \n\n")
    }
    x <- model.matrix(formula, mframe)  # features
    N <- nrow(y)  # number of data points
    J <- ncol(y)  # number of categories
    K.j <- t(matrix(apply(y, 2, max)))  # number of outcomes for each category
    R <- nclass
    S <- ncol(x)  # number of features
    eflag <- FALSE  # set to TRUE if find an error
    probs.start.ok <- TRUE
    ret <- list()  # list of items to return

    # nclass == 1 will use original code
    # poLCAParallel edits the orginial code for the nclass > 1 case
    if (R == 1) {
        ret$probs <- list()
        for (j in 1: J) {
            ret$probs[[j]] <- matrix(NA, nrow=1, ncol=K.j[j])
            for (k in 1: K.j[j]) {
                ret$probs[[j]][k] <- sum(y[, j] == k) / sum(y[, j] > 0)
            }
        }
        ret$probs.start <- ret$probs
        ret$P <- 1
        prior <- matrix(1, nrow=N, ncol=1)
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
                if (sum(sapply(probs.start,dim)[1, ] == R) != J) {
                    probs.start.ok <- FALSE
                }
                if (sum(sapply(probs.start,dim)[2, ] == K.j) != J) {
                    probs.start.ok <- FALSE
                }
                if (sum(round(sapply(probs.start, rowSums), 4) == 1)
                        != (R*J)) {
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
            initial_prob_vector <- c(initial_prob_vector,
                                    initial_prob[[1]]$vecprobs)
            irep <- irep + 1
        }

        # then generate random probabilities
        if (nrep > 1 | irep == 1) {
            for (repl in irep: nrep) {
                probs <- list()
                for (j in 1: J) {
                    probs[[j]] <- matrix(
                        runif(R*K.j[j]), nrow=R, ncol=K.j[j])
                    probs[[j]] <- probs[[j]] / rowSums(probs[[j]])
                }
                initial_prob[[repl]] <- poLCAParallel.vectorize(probs)
                initial_prob_vector <- c(initial_prob_vector,
                                            initial_prob[[repl]]$vecprobs)
            }
        }

        # random seed required to generate new initial values when needed
        seed <- sample.int(
            as.integer(.Machine$integer.max), 5, replace=TRUE)
        # run C++ code here, extract results
        em_results <- EmAlgorithmRcpp(x,
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
                                      seed)
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
            se <- poLCA.se(y, x, poLCAParallel.unvectorize(vp),
                            prior, rgivy)
            rownames(se$b) <- colnames(x)
        } else {
            se <- list(probs=NA, P=NA, b=matrix(nrow=S, ncol=R-1), var.b=NA)
        }

        # labelling b
        if (S > 1) {
            b <- matrix(b, nrow=S)
            rownames(b) <- colnames(x)
        } else {
            b <- NA
            se.b <- NA
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

        ret$coeff <- b  # coefficient estimates (when estimated)
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
    ret$npar <- (R*sum(K.j-1)) + (R-1)
    if (S>1) {
        ret$npar <- ret$npar + (S*(R-1)) - (R-1)
    }
    # Akaike Information Criterion
    ret$aic <- (-2 * ret$llik) + (2 * ret$npar)
    # Schwarz-Bayesian Information Criterion
    ret$bic <- (-2 * ret$llik) + (log(N) * ret$npar)
    # number of fully observed cases (if na.rm=F)
    ret$Nobs <- sum(rowSums(y == 0) == 0)

    y[y==0] <- NA
    ret$y <- data.frame(y)  # outcome variables
    ret$x <- data.frame(x)  # covariates, if specified
    for (j in 1: J) {
        rownames(ret$probs[[j]]) <- paste("class ", 1: R, ": ", sep="")
        if (is.factor(data[,match(colnames(y), colnames(data))[j]])) {
            lev <- levels(data[, match(colnames(y), colnames(data))[j]])
            colnames(ret$probs[[j]]) <- lev
            ret$y[, j] <- factor(ret$y[, j], labels=lev)
        } else {
            colnames(ret$probs[[j]]) <-
                paste("Pr(", 1: ncol(ret$probs[[j]]), ")", sep="")
        }
    }
    ret$N <- N  # number of observations

    # if no rows are fully observed or chi squared not requested
    if ((all(rowSums(y == 0) > 0)) | !calc.chisq) {
        ret$Chisq <- NA
        ret$Gsq <- NA
        ret$predcell <- NA
    } else {
        ret <- poLCAParallel.goodnessfit(ret)
    }

    ret$maxiter <- maxiter  # maximum number of iterations specified by user
    # number of residual degrees of freedom
    ret$resid.df <- min(ret$N, (prod(K.j)-1))-ret$npar
    class(ret) <- "poLCA"
    if (graphs) {
        plot.poLCA(ret)
    }
    if (verbose) {
        print.poLCA(ret)
    }
    ret$time <- Sys.time()-starttime  # how long it took to run the model
    ret$call <- match.call()
    return(ret)
}
