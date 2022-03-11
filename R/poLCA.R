poLCA <-
function(formula,data,nclass=2,maxiter=1000,graphs=FALSE,tol=1e-10,
                na.rm=TRUE,probs.start=NULL,nrep=1,verbose=TRUE,
                calc.se=FALSE, calc.chisq=TRUE, n.thread=detectCores()) {
    cat("\nUsing parallel version of poLCA\n")
    starttime <- Sys.time()
    mframe <- model.frame(formula,data,na.action=NULL)
    mf <- model.response(mframe)
    if (any(mf<1,na.rm=TRUE) | any(round(mf) != mf,na.rm=TRUE)) {
        cat("\n ALERT: some manifest variables contain values that are not
    positive integers. For poLCA to run, please recode categorical
    outcome variables to increment from 1 to the maximum number of
    outcome categories for each variable. \n\n")
        ret <- NULL
    } else {
    data <- data[rowSums(is.na(model.matrix(formula,mframe)))==0,]
    if (na.rm) {
        mframe <- model.frame(formula,data)
        y <- model.response(mframe)
    } else {
        mframe <- model.frame(formula,data,na.action=NULL)
        y <- model.response(mframe)
        y[is.na(y)] <- 0
    }
    if (any(sapply(lapply(as.data.frame(y),table),length)==1)) {
        y <- y[,!(sapply(apply(y,2,table),length)==1)]
        cat("\n ALERT: at least one manifest variable contained only one
    outcome category, and has been removed from the analysis. \n\n")
    }
    x <- model.matrix(formula,mframe)
    N <- nrow(y)
    J <- ncol(y)
    K.j <- t(matrix(apply(y,2,max)))
    R <- nclass
    S <- ncol(x)
    if (S>1) { calc.se <- TRUE }
    eflag <- FALSE
    probs.start.ok <- TRUE
    ret <- list()
    if (R==1) {
        ret$probs <- list()
        for (j in 1:J) {
            ret$probs[[j]] <- matrix(NA,nrow=1,ncol=K.j[j])
            for (k in 1:K.j[j]) { ret$probs[[j]][k] <- sum(y[,j]==k)/sum(y[,j]>0) }
        }
        ret$probs.start <- ret$probs
        ret$P <- 1
        ret$posterior <- ret$predclass <- prior <- matrix(1,nrow=N,ncol=1)
        ret$llik <- sum(log(poLCA.ylik.C(poLCA.vectorize(ret$probs),y)))
        if (calc.se) {
            se <- poLCA.se(y,x,ret$probs,prior,ret$posterior)
            ret$probs.se <- se$probs           # standard errors of class-conditional response probabilities
            ret$P.se <- se$P                   # standard errors of class population shares
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
        if (S>1) {
            cat("\n ALERT: covariates not allowed when nclass=1; will be ignored. \n \n")
            S <- 1
        }
    } else {
        if (!is.null(probs.start)) { # error checking on user-inputted probs.start
            if ((length(probs.start) != J) | (!is.list(probs.start))) {
                probs.start.ok <- FALSE
            } else {
                if (sum(sapply(probs.start,dim)[1,]==R) != J) probs.start.ok <- FALSE
                if (sum(sapply(probs.start,dim)[2,]==K.j) != J) probs.start.ok <- FALSE
                if (sum(round(sapply(probs.start,rowSums),4)==1) != (R*J)) probs.start.ok <- FALSE
            }
        }

        initial_prob <- list();
        initial_prob_vector = c();
        irep = 1;
        if (probs.start.ok & !is.null(probs.start)) {
            initial_prob[[1]] <- poLCAParallel.vectorize(probs.start)
            initial_prob_vector = c(initial_prob_vector, initial_prob[[1]]$vecprobs)
            irep = irep + 1
        }

        if (nrep > 1 | irep == 1) {
            for (repl in irep:nrep) {
                probs <- list()
                for (j in 1:J) {
                    probs[[j]] <- matrix(runif(R*K.j[j]),nrow=R,ncol=K.j[j])
                    probs[[j]] <- probs[[j]]/rowSums(probs[[j]])
                }
                initial_prob[[repl]] <- poLCAParallel.vectorize(probs)
                initial_prob_vector = c(initial_prob_vector, initial_prob[[repl]]$vecprobs)
            }
        }

        seed = sample.int(as.integer(.Machine$integer.max), 5, replace=TRUE)
        emResults <- EmAlgorithmRcpp(x,
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
        rgivy = emResults[[1]]
        prior = emResults[[2]]
        estimated_prob = emResults[[3]]
        b = emResults[[4]]
        ret$attempts = emResults[[5]]
        best_rep_index = emResults[[6]]
        nIter = emResults[[7]]
        best_initial_prob = emResults[[8]]
        eflag = emResults[[9]]

        lnL = ret$attempts[best_rep_index]

        vp = initial_prob[[1]]
        vp$vecprobs = estimated_prob

        ret$probs.start = initial_prob[[1]]
        ret$probs.start$vecprobs = best_initial_prob

        if (calc.se) {
            se <- poLCA.se(y,x,poLCAParallel.unvectorize(vp),prior,rgivy)
        } else {
            se <- list(probs=NA,P=NA,b=NA,var.b=NA)
        }

        ret$llik <- lnL             # maximum value of the log-likelihood
        ret$probs.start <- poLCAParallel.unvectorize(ret$probs.start)  # starting values of class-conditional response probabilities
        ret$probs <- poLCAParallel.unvectorize(vp) # estimated class-conditional response probabilities
        ret$probs.se <- se$probs           # standard errors of class-conditional response probabilities
        ret$P.se <- se$P                   # standard errors of class population shares
        ret$posterior <- rgivy             # NxR matrix of posterior class membership probabilities
        ret$predclass <- apply(ret$posterior,1,which.max)   # Nx1 vector of predicted class memberships, by modal assignment
        ret$P <- colMeans(ret$posterior)   # estimated class population shares
        ret$numiter <- nIter              # number of iterations until reaching convergence
        ret$probs.start.ok <- probs.start.ok # if starting probs specified, logical indicating proper entry format
        if (S>1) {
            b <- matrix(b,nrow=S)
            rownames(b) <- colnames(x)
            rownames(se$b) <- colnames(x)
            ret$coeff <- b                 # coefficient estimates (when estimated)
            ret$coeff.se <- se$b           # standard errors of coefficient estimates (when estimated)
            ret$coeff.V <- se$var.b        # covariance matrix of coefficient estimates (when estimated)
        } else {
            ret$coeff <- NA
            ret$coeff.se <- NA
            ret$coeff.V <- NA
        }
        ret$eflag <- eflag                 # error flag, true if estimation algorithm ever needed to restart with new initial values
    }
    names(ret$probs) <- colnames(y)
    if (calc.se) { names(ret$probs.se) <- colnames(y) }
    ret$npar <- (R*sum(K.j-1)) + (R-1)                  # number of degrees of freedom used by the model (number of estimated parameters)
    if (S>1) { ret$npar <- ret$npar + (S*(R-1)) - (R-1) }
    ret$aic <- (-2 * ret$llik) + (2 * ret$npar)         # Akaike Information Criterion
    ret$bic <- (-2 * ret$llik) + (log(N) * ret$npar)    # Schwarz-Bayesian Information Criterion
    ret$Nobs <- sum(rowSums(y==0)==0)                   # number of fully observed cases (if na.rm=F)

    y[y==0] <- NA
    ret$y <- data.frame(y)             # outcome variables
    ret$x <- data.frame(x)             # covariates, if specified
    for (j in 1:J) {
        rownames(ret$probs[[j]]) <- paste("class ",1:R,": ",sep="")
        if (is.factor(data[,match(colnames(y),colnames(data))[j]])) {
            lev <- levels(data[,match(colnames(y),colnames(data))[j]])
            colnames(ret$probs[[j]]) <- lev
            ret$y[,j] <- factor(ret$y[,j],labels=lev)
        } else {
            colnames(ret$probs[[j]]) <- paste("Pr(",1:ncol(ret$probs[[j]]),")",sep="")
        }
    }
    ret$N <- N                         # number of observations

    if ((all(rowSums(y==0)>0)) | !calc.chisq) {  # if no rows are fully observed or chi squared not requested
        ret$Chisq <- NA
        ret$Gsq <- NA
        ret$predcell <- NA
    } else {
        ret = poLCAParallel.goodnessfit(ret)
    }


    ret$maxiter <- maxiter             # maximum number of iterations specified by user
    ret$resid.df <- min(ret$N,(prod(K.j)-1))-ret$npar # number of residual degrees of freedom
    class(ret) <- "poLCA"
    if (graphs) plot.poLCA(ret)
    if (verbose) print.poLCA(ret)
    ret$time <- Sys.time()-starttime   # how long it took to run the model
    }
    ret$call <- match.call()
    return(ret)
}
