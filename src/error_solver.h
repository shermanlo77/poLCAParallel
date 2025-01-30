// poLCAParallel
// Copyright (C) 2024 Sherman Lo

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#ifndef POLCAPARALLEL_SRC_ERROR_SOLVER_H
#define POLCAPARALLEL_SRC_ERROR_SOLVER_H

#include <span>

#include "RcppArmadillo.h"

namespace polca_parallel {

/**
 * Abstract class used to work out and assign the standard errors
 *
 * Abstract class used to work out and assign the standard errors given the
 * score design matrix and the Jacobian matrix. Used by the StandardError class
 * and their derivatives, see standard_error.h.
 *
 * Pass the properties of the dataset and pointers to save the resulting errors
 * to the constructor. Then call Solve(), passing the score and Jacobian
 * matrices to calculate the standard errors (and covariance where appropriate)
 * and save it at the provided pointers.
 *
 * Derived classes are to implement the method Solve() to work out the standard
 * errors and save it, eg using eigen decomposition, SVD, inv(), pinv(), ...etc
 */
class ErrorSolver {
 protected:
  /** Number of data points, ie height of the score matrix */
  const std::size_t n_data_;
  /** Number of features */
  const std::size_t n_feature_;
  /** Sum of n_outcomes */
  const std::size_t sum_outcomes_;
  /** Number of clusters fitted */
  const std::size_t n_cluster_;
  /**
   * The size of the information matrix
   *
   * This is the same as the width of the score matrix and the height of the
   * Jacobian matrix
   */
  const std::size_t info_size_;
  /** The width of the Jacobian matrix */
  const std::size_t jacobian_width_;
  /**
   * Vector containing the standard error for the prior probabilities for each
   * cluster
   */
  std::span<double> prior_error_;
  /**
   * Vector containing the standard error for the outcome probabilities category
   * and cluster
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  std::span<double> prob_error_;
  /** Covariance matrix of the regression coefficient */
  arma::Mat<double> regress_coeff_error_;

 public:
  /**
   * Constructs an ErrorSolver
   *
   * Pass the properties of the dataset and pointers to save the resulting
   * errors to the constructor. Then call Solve(), passing the score and
   * Jacobian matrices to calculate the standard errors (and covariance where
   * appropriate) and save it at the provided pointers.
   *
   * @param n_data Number of data points
   * @param n_feature Number of features, required to be 1
   * @param sum_outcomes Sum of all integers in n_outcomes
   * @param n_cluster Number of clusters fitted
   * @param info_size The size of the information matrix
   * @param jacobian_width The width of the Jacobian matrix
   * @param prior_error Vector to contain the standard error for the prior
   * probabilities for each cluster, modified after calling Solve()
   * @param prob_error Vector to contain the standard error for the outcome
   * probabilities category and cluster, modified after calling Solve()
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param regress_coeff_error Matrix to contain the covariance matrix of the
   * regression coefficient, modified after calling Solve(). May not be used if
   * used in a non-regression setting.
   */
  ErrorSolver(std::size_t n_data, std::size_t n_feature,
              std::size_t sum_outcomes, std::size_t n_cluster,
              std::size_t info_size, std::size_t jacobian_width,
              std::span<double> prior_error, std::span<double> prob_error,
              std::span<double> regress_coeff_error);
  /**
   * Solves equations to work out the standard error and saves it
   *
   * Solves equations to work out the standard error and saves it where the
   * member variables prior_error, prob_error regress_coeff_error are pointing
   * to.
   *
   * @param score score matrix with the following dimensions
   * <ul>
   *   <li>dim 0: size n_data</li>
   *   <li>dim 1: size info_size</li>
   * </ul>
   * @param jacobian Jacobian matrix with the following dimensions
   * <ul>
   *   <li>dim 0: size info_size</li>
   *   <li>dim 1: size jacobian_width</li>
   * </ul>
   */
  virtual void Solve(const arma::Mat<double>& score,
                     const arma::Mat<double>& jacobian) = 0;
};

/**
 * Calculates standard errors from the eigencomposition of the info matrix
 *
 * Calculates the standard errors for the prior_error and prob_error. It
 * calculates the information matrix, from the score matrix, and inverts it. The
 * information matrix is typically ill-conditioned (very small positive and
 * negative eigenvalue), hence the justification for using an eigen
 * decomposition. As the same as pinv(), the inversion is done by inverting
 * the large eigenvalues and setting the small eigenvalues to zero. The root of
 * the inverted eigenvalues is taken so that the standard errors can be
 * obtained by the root column sum of squares.
 *
 * Note: the information matrix is calculated by S^T S where S is the score
 * matrix. This may cause numerical instability as S is commonly
 * ill-conditioned.
 *
 * Q = matrix containing columns of eigenvectors
 * D = diagonal of eigenvalues
 * J = jacobian matrix
 *
 * Eigendecomposition is given as = Q D Q^T
 *
 * The covariance of interest is J^T Q D^{-1} Q^T J
 *
 * For standard errors, take root column sum of squares D^{-1/2} Q^T J
 */
class InfoEigenSolver : public polca_parallel::ErrorSolver {
 public:
  InfoEigenSolver(std::size_t n_data, std::size_t n_feature,
                  std::size_t sum_outcomes, std::size_t n_cluster,
                  std::size_t info_size, std::size_t jacobian_width,
                  std::span<double> prior_error, std::span<double> prob_error,
                  std::span<double> regress_coeff_error);

  void Solve(const arma::Mat<double>& score,
             const arma::Mat<double>& jacobian) override;

 protected:
  /**
   * Extract errors of interest from eigen calculations
   *
   * Extract errors of interest given the eigenvectors and inverse eigenvalues
   * of the information matrix. Saves them to the member variables such as
   * prior_error, prob_error and regress_coeff_error
   *
   * @param eigval_inv the inverse of the eigenvalues of the information matrix
   * @param eigven eigenvectors of the information matrix
   * @param jacobian the jacobian matrix
   */
  virtual void ExtractErrorGivenEigen(const arma::Col<double>& eigval_inv,
                                      const arma::Mat<double>& eigvec,
                                      const arma::Mat<double>& jacobian);
};

/**
 * Calculates standard errors from the eigencomposition of the info matrix
 *
 * Calculates the standard errors for the prior_error, prob_error.and
 * regress_coeff_error. It calculates the information matrix, from the score
 * matrix, and inverts it. The information matrix is typically ill-conditioned
 * (very small positive and negative eigenvalue), hence the justification for
 * using an eigen decomposition. As the same as pinv(), the inversion is done by
 * inverting the large eigenvalues and setting the small eigenvalues to zero.
 * The root of the inverted eigenvalues is taken so that the standard errors
 * can be obtained by the root column sum of squares.
 *
 * The covariance matrix of the regression coefficients can be obtained directly
 * from the inverted information matrix, ie submatrix.
 *
 * Note: the information matrix is calculated by S^T S where S is the score
 * matrix. This may cause numerical instability as S is commonly
 * ill-conditioned.
 *
 * Q = matrix containing columns of eigenvectors
 * D = diagonal of eigenvalues
 * J = jacobian matrix
 *
 * Eigendecomposition is given as = Q D Q^T
 *
 * The covariance of interest is J^T Q D^{-1} Q^T J
 *
 * For standard errors, take root column sum of squares D^{-1/2} Q^T J
 *
 * For the regression coefficients covariance, take the top left (ie sub-matrix)
 * of the covariance of interest
 */
class InfoEigenRegressSolver : public polca_parallel::InfoEigenSolver {
 public:
  InfoEigenRegressSolver(std::size_t n_data, std::size_t n_feature,
                         std::size_t sum_outcomes, std::size_t n_cluster,
                         std::size_t info_size, std::size_t jacobian_width,
                         std::span<double> prior_error,
                         std::span<double> prob_error,
                         std::span<double> regress_coeff_error);

 protected:
  void ExtractErrorGivenEigen(const arma::Col<double>& eigval_inv,
                              const arma::Mat<double>& eigvec,
                              const arma::Mat<double>& jacobian) override;
};

/**
 * Calculates standard errors from the eigencomposition of the info matrix
 *
 * Calculates the standard errors for the prior_error and prob_error. It does an
 * SVD decomposition of the score matrix, which is typically ill-conditioned
 * with very small positive (and sometimes value zero) singular values. As with
 * pinv(), the inversion is done by inverting the large singular values and
 * setting the small singular values to zero. The standard errors can be
 * obtained by the root column sum of squares.
 *
 * The covariance matrix of the regression coefficients can be obtained directly
 * from the inverted information matrix, ie submatrix.
 *
 * This is supposed to be more numerically stable as it avoids doing S^T S
 * calculation. Benchmark vs InfoEigneSolver varies depending on the size of
 * S and perhaps more.
 *
 * S = score matrix (size n x p)
 * U = left orthogonal matrix (not needed) (size n x n)
 * V = right orthogonal matrix (size p x p)
 * D = diagonal matrix containing singular values (size n x p)
 * J = jacobian matrix
 *
 * S = U D V^T
 *
 * The covariance of interest is J^T (S^T S) ^{-1} J = J^T V D^{-2} V J
 *
 * For standard errors, take root column sum of squares D^{-1} Q^T J
 */
class ScoreSvdSolver : public polca_parallel::ErrorSolver {
 public:
  ScoreSvdSolver(std::size_t n_data, std::size_t n_feature,
                 std::size_t sum_outcomes, std::size_t n_cluster,
                 std::size_t info_size, std::size_t jacobian_width,
                 std::span<double> prior_error, std::span<double> prob_error,
                 std::span<double> regress_coeff_error);

  void Solve(const arma::Mat<double>& score,
             const arma::Mat<double>& jacobian) override;

  /**
   * Extract errors of interest from the SVD
   *
   * Extract errors of interest given the SVD of the score matrix. Saves them to
   * the member variables such as prior_error, prob_error and
   * regress_coeff_error
   *
   * @param singular_inv the inverse of the eigenvalues of the information
   * matrix
   * @param v_mat eigenvectors of the information matrix
   * @param jacobian the jacobian matrix
   */
  virtual void ExtractErrorGivenEigen(const arma::Col<double>& singular_inv,
                                      const arma::Mat<double>& v_mat,
                                      const arma::Mat<double>& jacobian);
};

/**
 * Calculates standard errors from the eigencomposition of the info matrix
 *
 * Calculates the standard errors for the prior_error, prob_error and
 * regress_coeff_error. It does an SVD decomposition of the score matrix, which
 * is typically ill-conditioned with very small positive (and sometimes value
 * zero) singular values. As with pinv(), the inversion is done by inverting the
 * large singular values and setting the small singular values to zero. The
 * standard errors can be obtained by the root column sum of squares.
 *
 * This is supposed to be more numerically stable as it avoids doing S^T S
 * calculation. Benchmark vs InfoEigneSolver varies depending on the size of
 * S and perhaps more.
 *
 * S = score matrix (size n x p)
 * U = left orthogonal matrix (not needed) (size n x n)
 * V = right orthogonal matrix (size p x p)
 * D = diagonal matrix containing singular values (size n x p)
 * J = jacobian matrix
 *
 * S = U D V^T
 *
 * The covariance of interest is J^T (S^T S) ^{-1} J = J^T V D^{-2} V J
 *
 * For standard errors, take root column sum of squares D^{-1} Q^T J
 *
 * For the regression coefficients covariance, take the top left (ie sub-matrix)
 * of the covariance of interest
 */
class ScoreSvdRegressSolver : public polca_parallel::ScoreSvdSolver {
 public:
  ScoreSvdRegressSolver(std::size_t n_data, std::size_t n_feature,
                        std::size_t sum_outcomes, std::size_t n_cluster,
                        std::size_t info_size, std::size_t jacobian_width,
                        std::span<double> prior_error,
                        std::span<double> prob_error,
                        std::span<double> regress_coeff_error);

 protected:
  void ExtractErrorGivenEigen(const arma::Col<double>& singular_inv,
                              const arma::Mat<double>& v_mat_t,
                              const arma::Mat<double>& jacobian) override;
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_ERROR_SOLVER_H
