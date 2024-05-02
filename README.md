# poLCAParallel

## Polytomous Variable Latent Class Analysis

### With Bootstrap Likelihood Ratio Test

Sherman E. Lo, Queen Mary, University of London

A reimplementation of poLCA
\[[CRAN](https://cran.r-project.org/web/packages/poLCA/index.html),
[GitHub](https://github.com/dlinzer/poLCA)\] in C++. It tries to reproduce
results and be as similar as possible to the original code but runs faster,
especially with multiple repetitions by using multiple threads.

## About poLCAParallel

The package poLCAParallel reimplements poLCA fitting, standard error
calculations, goodness of fit tests and the bootstrap log-likelihood ratio test
in C++. This was done using [Rcpp](https://cran.r-project.org/web/packages/Rcpp)
and [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo) which
allows R to run fast C++ code. Additional notes include:

* The API remains the same as the original poLCA with a few additions
* It tries to reproduce results from the original poLCA
* The code uses [Armadillo](http://arma.sourceforge.net/) for linear algebra
* Multiple repetitions are done in parallel using
  [`std::thread`](https://www.cplusplus.com/reference/thread/) for multi-thread
  programming and [`std::mutex`](https://www.cplusplus.com/reference/mutex/) to
  prevent data races
* Direct inversion of matrices is avoided to improve numerical stability and
  performance
* Response probabilities are reordered to increase cache efficiency
* Use of [`std::map`](https://en.cppreference.com/w/cpp/container/map) for the
  chi-squared calculations to improve performance

Further reading is available on a
[QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages.html).

## About poLCA

poLCA is a software package for the estimation of latent class models and latent
class regression models for polytomous outcome variables, implemented in the R
statistical computing environment.

Latent class analysis (also known as latent structure analysis) can be used to
identify clusters of similar "types" of individuals or observations from
multivariate categorical data, estimating the characteristics of these latent
groups, and returning the probability that each observation belongs to each
group. These models are also helpful in investigating sources of confounding and
nonindependence among a set of categorical variables, as well as for density
estimation in cross-classification tables. Typical applications include the
analysis of opinion surveys; rater agreement; lifestyle and consumer choice; and
other social and behavioral phenomena.

The basic latent class model is a finite mixture model in which the component
distributions are assumed to be multi-way cross-classification tables with all
variables mutually independent. The model stratifies the observed data by a
theoretical latent categorical variable, attempting to eliminate any spurious
relationships between the observed variables. The latent class regression model
makes it possible for the researcher to further estimate the effects of
covariates (or "concomitant" variables) on predicting latent class membership.

poLCA uses expectation-maximization and Newton-Raphson algorithms to find
maximum likelihood estimates of the parameters of the latent class and latent
class regression models.

## Recommended Prerequisites

The following prerequisites are recommended to be installed:

* R packages for installing and compiling:
  * [devtools](https://cran.r-project.org/web/packages/devtools/index.html)
  * [Rcpp](https://cran.r-project.org/web/packages/Rcpp)
  * [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo)
  * [roxygen2](https://cran.r-project.org/web/packages/roxygen2/index.html)
* Dependent R packages:
  * [MASS](https://cran.r-project.org/web/packages/MASS/index.html)
  * [parallel](https://www.rdocumentation.org/packages/parallel/)
  * [poLCA](https://cran.r-project.org/web/packages/poLCA/index.html)
  * [scatterplot3d](https://cran.r-project.org/web/packages/scatterplot3d/)

## Recommended Installation Instructions

The easiest way to install poLCAParallel is to use R with
[devtools](https://cran.r-project.org/web/packages/devtools/index.html).

### Install From GitHub

Run the following in R

```r
devtools::install_github("QMUL/poLCAParallel@package")
```

### Install From Releases

Download the `.zip` or `.tar.gz` file from the releases. Install it in R using

```r
devtools::install_local(<PATH TO .zip OR .tar.gz FILE>)
```

## User's Notes

### Citation

Please consider citing the corresponding
[QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages.html)

* Lo, S.E. (2022). Speeding up and Parallelising R packages (using Rcpp and C++)
  | QMUL ITS Research Blog.
  [[link]](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages.html)

and the publication below which this software was originally created for

* Eto F, Samuel M, Henkin R, Mahesh M, Ahmad T, et al. (2023). Ethnic
  differences in early onset multimorbidity and associations with health service
  use, long-term prescribing, years of life lost, and mortality: A
  cross-sectional study using clustering in the UK Clinical Practice Research
  Datalink. *PLOS Medicine,* 20(10): e1004300.
  <https://doi.org/10.1371/journal.pmed.1004300>

### Tips

* When using `model <- poLCAParallel::poLCA()`, set the parameters
  `calc.se=FALSE` and `calc.chisq=FALSE` to avoid doing standard error and
  goodness of fit calculations respectively. This will save time if you do not
  require those results. You can always calculate them afterwards using
  `model <- poLCAParallel::poLCAParallel.se(model)` and
  `model <- poLCAParallel::poLCAParallel.goodnessfit(model)`.
* Make use of multiple repetitions and threads. When using
  `poLCAParallel::poLCA()`, set `nrep=1` to do a test run and gauge how long it
  takes. Afterwards, set `nrep` to a bigger number to try different initial
  values in parallel.
* When using `poLCAParallel::poLCA()`, set `n.thread` to set the number of
  threads to be used by the computer. By default, it uses all detectable
  threads.
* There is an experimental option to use Laplace smoothing on the response
  probabilities when doing standard error calculations. This provides better
  numerical stability and avoids very small standard errors. To use it, either
  * In `poLCAParallel::poLCA()`, set `se.smooth=TRUE`
  * Or in `poLCAParallel::poLCAParallel.se()`, set `is_smooth=TRUE`
* When using the regression model, it is encouraged to normalise your data frame
  to provide better numerical stability.
* Use `set.seed()` before using `poLCAParallel::poLCA()` to set the seed for
  random number generation. This ensures reproducibility when reporting what
  seed you have used.

### Example Code

R scripts which compare poLCAParallel with poLCA are provided in `exec/`.
An example use of a bootstrap likelihood ratio test is shown in `exec/3_blrt.R`.

### Changes from the Original Code

* In `poLCAParallel::poLCA()`, the following arguments have been added:
  * `n.thread` is provided to specify the number of threads to use.
  * `calc.chisq` is provided to specify if you want to conduct goodness of fit
    tests or not.
  * `se.smooth` is provided if you wish to use Laplace smoothing on the response
    probabilities in the standard error calculations.
* The prior probabilites is a return value, accessible with `$prior`.
* The stopping condition of the EM algorithm has changed slightly. If the
  log-likelihood change after an iteration of EM is too small, the stopping
  condition is evaluated after the E step rather than the M step. This is so
  that the by-product of the E step is reused when calculating the
  log-likelihood.
* The Newton step uses a linear solver rather than directly inverting the
  Hessian matrix in the regression model.
* The output `probs.start` are the initial probabilities used to achieve the
  maximum log-likelihood from *any* repetition rather than from the first
  repetition.
* The output `eflag` is set to `TRUE` if *any* repetition has to be restarted,
  rather than the repetition which achieves maximum log-likelihood.
* The standard error is not calculated if `calc.se` is set to `FALSE` even in
  poLCA regression. Previously, the standard error is calculated regardless of
  `calc.se` in poLCA regression.
* In the standard error calculations, an SVD is done on the score matrix,
  rather than inverting the information matrix.
* Any errors in the input data will call `stop()` rather than return a `NULL`.
* No rounding in the return value `predcell`.

## Developer's Notes

### Installing as a Developer

The following installation instructions are useful if you wish to develop the
code and install a locally modified version of the package. The instructions do
not require the R package devtools.

Requires the R packages for compiling:

* [Rcpp](https://cran.r-project.org/web/packages/Rcpp)
* [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo)
* [roxygen2](https://cran.r-project.org/web/packages/roxygen2/index.html)

Requires the dependent R packages:

* [MASS](https://cran.r-project.org/web/packages/MASS/index.html)
* [parallel](https://www.rdocumentation.org/packages/parallel/)
* [poLCA](https://cran.r-project.org/web/packages/poLCA/index.html)
* [scatterplot3d](https://cran.r-project.org/web/packages/scatterplot3d/)

Git clone this repository

```bash
git clone https://github.com/QMUL/poLCAParallel.git
```

Run the following to generate additional code and documentation so that the
package can be compiled correctly

```bash
R -e "Rcpp::compileAttributes('poLCAParallel')"
R -e "roxygen2::roxygenize('poLCAParallel')"
```

Install the package using

```bash
R CMD INSTALL --preclean --no-multiarch poLCAParallel
```

### Troubleshooting

If the installation instructions fail or there are other problems, please check
the possible following prerequisites are installed:

* A C++ compiler like [gcc](https://gcc.gnu.org/) or
  [clang](https://clang.llvm.org/)
* [Armadillo](http://arma.sourceforge.net/) (see the
  [installation manual](https://arma.sourceforge.net/download.html) for further
  details)
* [LAPACK](http://www.netlib.org/lapack/)
* [OpenBLAS](https://www.openblas.net/)

An [Apptainer](https://apptainer.org/) defintion file `poLCAParallel.def` is
provided which installs R, prerequisites and poLCAParallel in a container. This
may be useful for further troubleshooting.

### Development Notes

* When calculating the likelihood, probabilities are iteratively multiplied,
  this is much faster than taking the sum of log probabilities. However, to
  avoid underflow errors, the calculation of the likelihood uses the sum of log
  probabilities when an underflow is detected. See `PosteriorUnnormalize()` in
  `src/em_algorithm.*` for the implementation.
* In the standard error calculations, the score matrix is typically
  ill-conditioned. Consider pre-conditioning the matrix.
* In the poLCA regression model, consider using multiple Newton steps instead
  of one single step in the EM algorithm.

### Actions For The Next Major Version

The following should be actioned in the next major version:

* The R package MASS is not required as a prerequisite.

The following R functions (and their corresponding C functions if available) are
marked as deprecated and should be deleted in the next major version

* `poLCA.se()` - no longer needed, reimplemented in `poLCAParallel.se()`
* `poLCA.dLL2dBeta.C()` - no longer needed, reimplemented in
    `em_algorithm_regress.cc`
* `poLCA.probHat.C` - no longer needed, the goodness of fit test is
    reimplemented in `goodness_fit.cc`

### Code Style

All generated documents and codes, eg from

```bash
R -e "Rcpp::compileAttributes('poLCAParallel')"
```

and

```bash
R -e "roxygen2::roxygenize('poLCAParallel')"
```

shall not be included in the `master` branch. Instead, they shall be in the
`package` branch so that this package can be installed using
`devtools::install_github("QMUL/poLCAParallel@package")`. This is to avoid
having duplicate documentation and generated code on the `master` branch.

Semantic versioning is used and tagged. Tags on the `master` branch shall have
`v` prepended and `-master` appended, eg. `v1.1.0-master`. The corresponding
tag on the `package` branch shall only have `v` prepended, eg. `v1.1.0`.

There was an attempt to use the
[Google C++ style guide](https://google.github.io/styleguide/cppguide.html).

Armadillo objects are used sparingly, preferring the use of `double*` when
passing vectors and matrices.

### C++ Source Code Documentation

The C++ code documentation can be created with [Doxygen](https://doxygen.nl/)
by running

```console
doxygen
```

and viewed at `html/index.html`.

## References

* Bandeen-roche, K., Miglioretti, D. L., Zeger, S. L., and Rathouz, P. J.
  (1997). Latent variable regression for multiple discrete outcomes. *Journal of
  the American Statistical Association*, 92(440):1375–1386.
  [[link]](https://doi.org/10.1080/01621459.1997.10473658)
* Dziak, J. J., Lanza, S. T., & Tan, X. (2014). Effect size, statistical power,
  and sample size requirements for the bootstrap likelihood ratio test in latent
  class analysis. *Structural Equation Modeling: A Multidisciplinary Journal*,
  21(4):534-552.
  [[link]](https://www.tandfonline.com/doi/full/10.1080/10705511.2014.919819?casa_token=LgaSzKeeB8MAAAAA%3AB80XwZEIkLOIVsD4Gvp6O0gfktOnIqA6dOBBvUZIjjhs-7ilLIZJC_TmxCh8Umh45d0sWez4-em9)
* Linzer, D.A. & Lewis, J. (2013). poLCA: Polytomous Variable Latent
  Class Analysis. R package version 1.4.
  [[link]](https://github.com/dlinzer/poLCA)
* Linzer, D.A. & Lewis, J.B. (2011). poLCA: An R package for polytomous
  variable latent class analysis. *Journal of Statistical Software*,
  42(10): 1-29.
  [[link]](http://www.jstatsoft.org/v42/i10)

## License

The software is under the GNU GPL 2.0 license, as with the original poLCA code,
stated in their
[documentation](https://cran.r-project.org/web/packages/poLCA/index.html).
