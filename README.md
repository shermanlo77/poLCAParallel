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

The library poLCAParallel reimplements poLCA and the bootstrap log-likelihood
ratio test in C++. This was done using
[Rcpp](https://cran.r-project.org/web/packages/Rcpp) and
[RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo) which
allows C++ code to interact with R. Additional notes include:

* The code uses [Armadillo](http://arma.sourceforge.net/) for linear algebra
* Multiple repetitions are done in parallel using
  [`<thread>`](https://www.cplusplus.com/reference/thread/) for multi-thread
  programming and [`<mutex>`](https://www.cplusplus.com/reference/mutex/) to
  prevent data races
* Response probabilities are reordered to increase cache efficiency
* Use of [`std::map`](https://en.cppreference.com/w/cpp/container/map) for the
  chi-squared calculations

Further reading is available on a
[QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages.html).

## About the Original Code

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

## Recommend Prerequisites

If the recommended installation instructions fail or there are other problems,
please check the possible following prerequisites are installed:

* R packages for installing and compiling:
  * [devtools](https://cran.r-project.org/web/packages/devtools/index.html)
  * [Rcpp](https://cran.r-project.org/web/packages/Rcpp)
  * [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo)
* Dependent R packages:
  * [MASS](https://cran.r-project.org/web/packages/MASS/index.html)
  * [parallel](https://www.rdocumentation.org/packages/parallel/)
  * [scatterplot3d](https://cran.r-project.org/web/packages/scatterplot3d/)
* A C++ compiler like [gcc](https://gcc.gnu.org/)
* [Armadillo](http://arma.sourceforge.net/)
* [LAPACK](http://www.netlib.org/lapack/)
* [OpenBLAS](https://www.openblas.net/)

## Installation using R and devtools::install_github()

Requires the R package
[devtools](https://www.r-project.org/nosvn/pandoc/devtools.html)

Run the following in R

```r
devtools::install_github("QMUL/poLCAParallel@package")
```

## Installation from Releases

Requires the R packages [Rcpp](https://cran.r-project.org/web/packages/Rcpp),
[RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo) and
[scatterplot3d](https://cran.r-project.org/web/packages/scatterplot3d/index.html).

Download the `.tar.gz` file from the releases. Install it using using
`R CMD INSTALL`, for example

```bash
R CMD INSTALL --preclean --no-multiarch poLCAParallel-*.*.*.tar.gz
```

## Installation using a git clone

Requires the R packages [Rcpp](https://cran.r-project.org/web/packages/Rcpp) and
[RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo). The R
package [devtools](https://www.r-project.org/nosvn/pandoc/devtools.html) is
recommended. Optionally, creating the documentation requires the R package
[roxygen2](https://cran.r-project.org/web/packages/roxygen2/index.html).

Git clone this repository.

```bash
git clone https://github.com/QMUL/poLCAParallel.git
```

Run the following so that the package can be compiled correctly

```bash
R -e "Rcpp::compileAttributes('poLCAParallel')"
```

and optionally for the documentation

```bash
R -e "devtools::document('poLCAParallel')"
```

Finally

```bash
R -e "devtools::install('poLCAParallel')"
```

to install the package. Alternatively, `R CMD INSTALL` can be used as shown
below in a terminal

```bash
R CMD INSTALL --preclean --no-multiarch poLCAParallel
```

## Changes from the Orginal Code

R scripts which compare poLCAParallel with poLCA are provided in `exec/`.
Example use of a bootstrap likelihood ratio test is shown in `exec/3_blrt.R`.

* The stopping condition of the EM algorithm, if the log-likelihood change after
  an iteration of EM is too small, is evaluated after the E step rather than the
  M step. This is so that the by-product of the E step is reused when
  calculating the log-likelihood.
* The Newton step uses a linear solver rather than directly inverting the
  Hessian matrix.
* The output `probs.start` are the initial probabilities used to achieve the
  maximum log-likelihood from *any* repetition rather than from the first
  repetition.
* The output `eflag` is set to `TRUE` if *any* repetition has to be restarted,
  rather than the repetition which achieves maximum log-likelihood.
* An additional argument `n.thread` is provided to specify the number of threads
  to use.
* The standard error is not calculated if `calc.se` is set to `FALSE` even in
  poLCA regression. Previously, the standard error is calculated regardless of
  `calc.se` for poLCA regression.
* Any errors in the input data will call `stop()` rather than return a `NULL`.
* No rounding in the return value `predcell`.

## Further Development Notes

* The standard errors have not been implemented in C++. This is because there
  are possible problems with implementing the calculations of the standard
  error. They are discussed here [[.tex]](inst/note_standard_error.tex)
  [[.md]](inst/note_standard_error.md). In-built GitHub tools may not render
  equations correctly.
* When calculating the likelihood, probabilities are iteratively multiplied,
  this is much faster than taking the sum of log probabilities. However to avoid
  underflow errors, the calculation of the likelihood, instead, uses the sum of
  log probabilities when an underflow is detected. See `PosteriorUnnormalize()`
  in `src/em_algorithm.*` for the implementation.
* Multiple Newton steps can be taken instead of a single one.

## Code Style

All generated documents and codes, eg from

```bash
R -e "Rcpp::compileAttributes('poLCAParallel')"
```

and

```bash
R -e "devtools::document('poLCAParallel')"
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
handling vectors and matrices.

## C++ Source Code Documentation

The C++ code documentation can be created with [Doxygen](https://doxygen.nl/)
by running

```console
doxygen
```

and viewed at `html/index.html`.

## Citation

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

## References

* Dziak, J. J., Lanza, S. T., & Tan, X. (2014). Effect size, statistical power,
  and sample size requirements for the bootstrap likelihood ratio test in latent
  class analysis. *Structural Equation Modeling: A Multidisciplinary Journal*,
  21(4), 534-552.
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
