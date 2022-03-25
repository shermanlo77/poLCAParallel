# poLCAParallel
## Polytomous Variable Latent Class Analysis

Sherman E. Lo, Queen Mary, University of London

A reimplementation of poLCA \[[CRAN](https://cran.r-project.org/web/packages/poLCA/index.html), [GitHub](https://github.com/dlinzer/poLCA)\] in C++. It tries to reproduce results and be as similar as possible to the original code but runs faster, especially with multiple repetitions by using multiple threads.

## About the Original Code

poLCA is a software package for the estimation of latent class models and latent class regression models for polytomous outcome variables, implemented in the R statistical computing environment.

Latent class analysis (also known as latent structure analysis) can be used to identify clusters of similar "types" of individuals or observations from multivariate categorical data, estimating the characteristics of these latent groups, and returning the probability that each observation belongs to each group. These models are also helpful in investigating sources of confounding and nonindependence among a set of categorical variables, as well as for density estimation in cross-classification tables. Typical applications include the analysis of opinion surveys; rater agreement; lifestyle and consumer choice; and other social and behavioral phenomena.

The basic latent class model is a finite mixture model in which the component distributions are assumed to be multi-way cross-classification tables with all variables mutually independent. The model stratifies the observed data by a theoretical latent categorical variable, attempting to eliminate any spurious relationships between the observed variables. The latent class regression model makes it possible for the researcher to further estimate the effects of covariates (or "concomitant" variables) on predicting latent class membership.

poLCA uses expectation-maximization and Newton-Raphson algorithms to find maximum likelihood estimates of the parameters of the latent class and latent class regression models.

## Recommend Prerequisites

If the recommended installation instructions fail or there are other problems, please check the possible following prerequisites are installed:
* R packages:
    * [Rcpp](https://cran.r-project.org/web/packages/Rcpp)
    * [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo)
    * [MASS](https://cran.r-project.org/web/packages/MASS/index.html)
    * [parallel](https://www.rdocumentation.org/packages/parallel/)
    * [scatterplot3d](https://cran.r-project.org/web/packages/scatterplot3d/)
* A C++ compiler like [gcc](https://gcc.gnu.org/)
* [Armadillo](http://arma.sourceforge.net/)
* [LAPACK](http://www.netlib.org/lapack/)
* [OpenBLAS](https://www.openblas.net/)

## Installation using Linux Terminal

Requires the R packages [Rcpp](https://cran.r-project.org/web/packages/Rcpp) and [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo).

Git clone this repository.
In R,
```
Rcpp::compileAttributes("<path_to_poLCAParallel>")
```
then in a Linux terminal at the directory containing the repository
```
R CMD INSTALL --preclean --no-multiarch poLCAParallel
```

## Installation using RStudio

You may install it using RStudio by opening the `.Rproj` file and clicking on `Build -> Install and Restart`.

## About poLCAParallel

The library poLCAParallel reimplements poLCA in C++. This was done using [Rcpp](https://cran.r-project.org/web/packages/Rcpp) and [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo) which allows C++ code to interact with R. Addition notes include:
*  The code uses [Armadillo](http://arma.sourceforge.net/) for linear algebra
*  Multiple repetitions are done in parallel using [`<thread>`](https://www.cplusplus.com/reference/thread/) for multi-thread programming and [`<mutex>`](https://www.cplusplus.com/reference/mutex/) to prevent data races
*  Response probabilities are reordered to increase cache efficiency
*  Use of [`std::map`](https://en.cppreference.com/w/cpp/container/map) for the chi-squared calculations

## Changes from the Orginal Code

* The stopping condition of the EM algorithm, if the log-likelihood change after an iteration of EM is too small, is evaluated after the E step rather than the M step. This is so that the by-product of the E step is reused when calculating the log-likelihood.
* The Newton step uses a linear solver rather than directly inverting the Hessian matrix.
* The output `probs.start` are the initial probabilities used to achieve the maximum log-likelihood from *any* repetition rather than from the first repetition.
* The output `eflag` is set to `TRUE` if *any* repetition had to be restarted, rather than the repetition which achieves maximum log-likelihood.
* An additional argument `n.thread` is provided to specify the number of threads to use.
* The standard error is not calculated if `calc.se` is set to `FALSE` even in poLCA regression. Previously, the standard error is calculated regardless of `calc.se` for poLCA regression.
* Any errors in the input data will call `stop()` rather than return a `NULL`.
* No rounding in the return value `predcell`.

## Further Development Notes

* There are possible problems with implementing the calculations of the standard error. They are discussed [here](note_standard_error.tex).
* There is a possible underflow error if the number of categories is too large, more than ~300. This is because in the calculation of the log-likelihood, the probabilities from each category are multiplied by each other. If there are $J$ categories, then there are $J$ probabilities to multiply together. This is addressed in commit 85ee419 but reverted. Consider summing over log space instead.
* Multiple Newton steps can be taken instead of a single one.

## Code Style

There was an attempt to use the [Google C++ style guide](https://google.github.io/styleguide/cppguide.html). There are deviations including:
* Class methods are defined in a Java style
* Post-increments are preferred over pre-increments
* `public`, `protected` and `private` keywords have a 2 space indent
* The use of C style casting
* The use `sizeof(double)`

Armadillo objects are used sparingly, preferring the use of `double*` when handling vectors and matrices.

## Citation

Users of poLCA are requested to cite the software package as:

Linzer, Drew A. and Jeffrey Lewis. 2013. "poLCA: Polytomous Variable Latent Class Analysis." R package version 1.4. http://dlinzer.github.com/poLCA.

and

Linzer, Drew A. and Jeffrey Lewis. 2011. "poLCA: an R Package for Polytomous Variable Latent Class Analysis." Journal of Statistical Software. 42(10): 1-29. http://www.jstatsoft.org/v42/i10

## License

The software is under the GNU GPL 2.0 license, as with the original poLCA code, stated in their [documentation](https://cran.r-project.org/web/packages/poLCA/index.html).
