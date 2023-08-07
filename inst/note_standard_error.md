# On the Standard Error in `poLCA`

Sherman E Lo

This note describes the current implementation of the standard error in the `R`
package `poLCA`. The mathematics is explained in Linzer and Lewis (2011). The
observed information matrix is constructed, using the data and estimated
parameters, which is then used to solve a linear equation. Computationally,
there are flaws in the implementation especially when the observed information
matrix is, or close to, being singular.

The mathematical problem shall be explained here along with an explanation of
the `poLCA` implementation of it. The problems are discussed here along with
possible solutions and further discussions.

The notation tries to be as similar as possible to Linzer and Lewis (2011).

## Standard Error

The latent class model is discussed here, but can be extended to the latent
class regression models.

Suppose there are $N$ data points, each with log likelihood

$$
    \ln L_i = \sum_{i=1}^N \ln \sum_{r=1}^R p_r \prod_{j=1}^J \prod_{k=1}^{K_j}
              \pi_{j,r,k}^{Y_{i,j,k}}
$$

where there are $R$ clusters, $J$ categories, $K_1, K_2,\ldots,K_J$ are the
number of responses for each category, $p_r$ is the prior probability,
$\pi_{j,r,k}$ is the outcome probability and $Y_{i,j,k}$ is the observed
response for $i=1,2,\ldots,N$, $j=1=2,\ldots,J$, $r=1,2,\ldots,R$ and for a
given $j$, $k=1,2,\ldots,K_j$.

In addition, the parameters are re-parameterised such that

$$
    p_1 = \dfrac{1}{\sum_{r'=2}^R \exp(\omega_{r'})}
$$

$$
    p_r = \dfrac{\exp(\omega_{r'})}{\sum_{r'=2}^R \exp(\omega_{r'})}
$$

$$
    \pi_{j,r,1} = \dfrac{1}{\sum_{k'=2}^{K_j} \exp(\phi_{j,r,k'})}
$$

$$
    \pi_{j,r,k} = \dfrac{\exp(\phi_{j,r,k})}{\sum_{k'=2}^{K_j}
                  \exp(\phi_{j,r,k'})}
$$

The score vector $\mathbf{s}_i$ is

$$
    \mathbf{s}_i = \mathbf{s}_i(Y_i, \Psi) = \nabla_\Psi  \ln L_i
$$

where

$$
    \Psi =
    \begin{pmatrix}
        \omega_2 \\
        \vdots \\
        \omega_R \\
        \phi_{1,1,2} \\
        \vdots \\
        \phi_{1,1,K_1} \\
        \phi_{2,1,1} \\
        \vdots \\
        \phi_{J,1,K_J} \\
        \phi_{1,2,2} \\
        \vdots \\
        \phi_{J, R, K_J}
    \end{pmatrix}
$$

is a vector of all parameters. The scores are calculated to be

$$
    \mathbf{s}_i
    =
    \begin{pmatrix}
        \dfrac{\partial \ln L_i}{\partial \omega_2} \\
        \vdots \\
        \dfrac{\partial \ln L_i}{\partial \omega_R} \\
        \dfrac{\partial \ln L_i}{\partial \phi_{1,1,2}} \\
        \vdots \\
        \dfrac{\partial \ln L_i}{\partial \phi_{J,R,K_J}}
    \end{pmatrix}
    =
    \begin{pmatrix}
        \theta_{i2} - p_{2} \\
        \vdots \\
        \theta_{iR} - p_{R} \\
        \theta_{ir}(Y_{i,1,2} - \pi_{1,1,2}) \\
        \vdots \\
        \theta_{ir}(Y_{i,J,K_J} - \pi_{J,R,K_J}) \\
    \end{pmatrix}
$$

where $\theta_{i,r}$ are the posterior probabilities

$$
    \theta_{i,r} = \dfrac{
        p_r \prod_{j=1}^J \prod_{k=1}^{K_j} \pi_{j,r,k}^{Y_{i,j,k}}
    } {
        \sum_{r'=1}^R p_{r'} \prod_{j=1}^J \prod_{k=1}^{K_j}
             \pi_{j,r',k}^{Y_{i,j,k}}
    }
$$

The observed information matrix $\mathbf{I}$ is constructed using the collection
of the score vectors

$$
    \mathbf{I} = \sum_{i=1}^N \mathbf{s}_i \mathbf{s}_i^\textup{T}
$$

which can be written in matrix form

$$
    \mathbf{I} = \mathbf{S}^\textup{T} \mathbf{S}
$$

where $\mathbf{S}$ is the design matrix of score vectors

$$
    \mathbf{S} =
    \begin{pmatrix}
      \mathbf{s}_1^\textup{T} \\
      \mathbf{s}_2^\textup{T} \\
      \vdots \\
      \mathbf{s}_N^\textup{T}
    \end{pmatrix}
$$

The delta method can be written as

$$
    \Sigma = \mathbf{J}^\textup{T} \mathbf{I}^{-1} \mathbf{J}
$$
where $\mathbf{J}$ is the Jacobian matrix and has the form

$$
    \mathbf{J} =
    \begin{pmatrix}
        \mathbf{J}_\omega & 0 & 0 & \ldots & 0\\
        0 & \mathbf{J}_{\phi_{1,1}} & 0 & \cdots & 0\\
        0 & 0 & \mathbf{J}_{\phi_{2,1}} & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & \mathbf{J}_{\phi_{J,R}}
    \end{pmatrix}
$$

where

$$
    \mathbf{J}_\omega =
    \begin{pmatrix}
        -p_1 p_2 & p_2 (1-p_2) & -p_3 p_2 & \cdots & -p_R p_2 \\
        -p_1 p_3 & -p_2 p_3 & p_3 (1-p_3) & \cdots & -p_R p_3\\
        \vdots & \vdots  & \vdots & \ddots & \vdots \\
        -p_1 p_R & -p_2 p_R & -p_3 p_R & \cdots & p_R (1-p_R)
    \end{pmatrix}
$$

and

$$
    \mathbf{J}_{\phi_{j,r}} =
    \begin{pmatrix}
        -\pi_{j,r,1} \pi_{j,r,2} & \pi_{j,r,2} (1-\pi_{j,r,2})
            & -\pi_{j,r,3} \pi_{j,r,2} & \cdots & -\pi_{j,r,K_j} \pi_{j,r,2} \\
        -\pi_{j,r,1} \pi_{j,r,3}& -\pi_{j,r,2} \pi_{j,r,3}
            & \pi_{j,r,3} (1-\pi_{j,r,3}) & \cdots
            & -\pi_{j,r,K_j} \pi_{j,r,3}\\
        \vdots & \vdots  & \vdots & \ddots & \vdots \\
        -\pi_{j,r,1} \pi_{j,r,K_j} & -\pi_{j,r,2} \pi_{j,r,K_j}
            & -\pi_{j,r,3} \pi_{j,r,K_j} & \cdots
            & \pi_{j,r,K_j} (1-\pi_{j,r,K_j})
    \end{pmatrix}
$$

The standard errors of interest are the diagonal elements of
$\Sigma = \mathbf{J}^\textup{T} \mathbf{I}^{-1} \mathbf{J}$.

## Implementation

The package `poLCA` uses the pseudo-inverse to calculate $\mathbf{I}^{-1}$.
Calculating the inverse of a matrix is highly discouraged as it is known to be
inaccurate and slow. The current implementation also does the full matrix
multiplication $\mathbf{J}^\textup{T} \mathbf{I}^{-1} \mathbf{J}$, however, this
could be wasteful because only the diagonal elements are needed.

It should be faster to do a Cholesky decomposition of

$$
    \mathbf{I}=\mathbf{L}\mathbf{L}^\textup{T}
$$

where $\mathbf{L}$ is a lower triangular matrix. Then the linear equation
becomes

$$
    \Sigma = (\mathbf{L}^{-1}\mathbf{J})^\textup{T} \mathbf{L}^{-1} \mathbf{J}
$$

The standard errors can be obtained by taking the column sum of squares of
$\mathbf{L}^{-1} \mathbf{J}$ which can be solved using a lower triangular
solver.

However, it is not uncommon for $\mathbf{I}^{-1}$ to be (or close to) singular.
This can happen when the EM algorithm overfits especially when the posteriors
$\theta_{ir}$ and estimated probabilities $\pi_{j,r,k}$ take values of (or close
to) zero or one. This can cause $\theta_{ir}(Y_{i,r,k}-\pi_{j,r,k})=0$ for all
$i$ and make at least one row and column of $\mathbf{I}$ to be all zeros, thus
singular.

It was suspected that the use of the pseudo-inverse in `poLCA` was a way to
handle singular matrices. It, however, does not solve the problem from the
ground up as the covariance does need to be positive definite in the first place
for it to make sense mathematically.

In order for $\mathbf{I}$ to be Cholesky decomposable, it must be
positive-definite and, ideally, well-conditioned.

A positive-definite $\mathbf{I}$ can be achieved by preventing the posteriors
$\theta_{ir}$ and estimated probabilities $\pi_{j,r,k}$ to take values of zero
or one. Laplace smoothing may be appropriate which encourages $\theta_{ir}$ and
$\pi_{j,r,k}$ to move closer to 0.5, rather than zero or one. This can be done
by replacing the calculations of $\theta_{ir}$ and $\pi_{j,r,k}$ with

$$
    \tilde{\pi}_{j,r,k} = \dfrac{N\pi_{j,r,k} + 1}{N + K_j}
$$

and

$$
    \tilde{\theta}_{i,r} = \dfrac{N\theta_{i,r,k} + 1}{N + R}
$$

This would alter the calculations but should guarantee $\mathbf{I}$ to be
positive definite. It may also be applied to the prior for good measure

$$
    \tilde{p}_{r} = \dfrac{N p_{r} + 1}{N + R}
$$

However, $\mathbf{I}$ may not be well-conditioned and should be pre-conditioned
so that the values in $\mathbf{L}$ are sensible for the triangular solver. Let
$\mathbf{D}$ be a diagonal matrix with elements
$\sqrt{\text{diag}\left[\mathbf{I}\right]}$. Then the linear equation can be
written as

$$
    \Sigma = \mathbf{J}^\textup{T} \mathbf{D}
    \left(
        \mathbf{D}\mathbf{I}\mathbf{D}
    \right)^{-1} \mathbf{D} \mathbf{J}
$$

Instead, a Cholesky decomposition of $\mathbf{D}\mathbf{I}\mathbf{D} =
\mathbf{L}\mathbf{L}^\textup{T}$ can be taken, hoping it would have a better
condition number than $\mathbf{I}$ by itself. Then the standard error are the
column sum of squares of $\mathbf{L}^{-1} \left(\mathbf{D}\mathbf{J}\right)$.

## Further Reading

Bandeen-roche et al. (1997) goes into the algebra of these calculations a bit
further. Louis (1982) mentions on calculating the observed information matrix
when using the EM algorithm. Lastly, the bootstrap (Efron, 1979) is a numerical
way of calculating the standard error by resampling from the data.

## References

* Bandeen-roche, K., Miglioretti, D. L., Zeger, S. L., and Rathouz, P. J.
(1997). Latent variable regression for multiple discrete outcomes. *Journal of
the American Statistical Association*, 92(440):1375–1386.
* Efron, B. (1979). Bootstrap methods: Another look at the Jackknife. *The
Annals of Statistics*, 7(1):1–26.
* Linzer, D. A. and Lewis, J. B. (2011). poLCA: An R package for polytomous
variable latent class analysis. *Journal of Statistical Software*, 42(10):1–29.
* Louis, T. A. (1982). Finding the observed information matrix when using the EM
algorithm. *Journal of the Royal Statistical Society*. Series B
(Methodological), 44(2):226–233.
