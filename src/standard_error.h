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

#ifndef POLCAPARALLEL_SRC_STANDARD_ERROR_H
#define POLCAPARALLEL_SRC_STANDARD_ERROR_H

#include <memory>
#include <span>

#include "RcppArmadillo.h"
#include "error_solver.h"
#include "smoother.h"
#include "util.h"

namespace polca_parallel {

/**
 * For calculating the standard errors of the fitted poLCA parameters
 *
 * For calculating the standard errors of the fitted poLCA parameters such as
 * <ul>
 *   <li>The prior probabilities for each cluster/class</li>
 *   <li>The probabilities for each outcome, category and cluster</li>
 *   <li>The regression coefficients
 * </ul>
 *
 * How to use:
 * <ul>
 *   <li>Instantiate and pass the required parameters, this includes allocated
 *   memory to store the resulting standard errors</li>
 *   <li>Call the method Calc()</li>
 * </ul>
 */
class StandardError {
 protected:
  /**
   * Design matrix of responses, matrix containing outcomes/responses
   * for each category as integers 1, 2, 3, .... The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each category</li>
   * </ul>
   */
  arma::Mat<int> responses_;
  /**
   * Vector of probabilities for each category and response,
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  std::span<double> probs_;
  /**
   * Design matrix of prior probabilities, probability data point is in
   * cluster m NOT given responses after calculations, it shall be in matrix
   * form with dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  arma::Mat<double> prior_;
  /**
   * Design matrix of posterior probabilities (also called responsibility)
   * probability data point is in cluster m given responses
   * matrix
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  arma::Mat<double> posterior_;
  /** Number of data points */
  const std::size_t n_data_;
  /** Number of features */
  const std::size_t n_feature_;
  /** Vector of number of outcomes for each category */
  NOutcomes n_outcomes_;
  /** Number of clusters to fit */
  const std::size_t n_cluster_;
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
  std::span<double> regress_coeff_error_;
  /** The size of the information matrix*/
  std::size_t info_size_;
  /** The width of the Jacobian matrix*/
  std::size_t jacobian_width_;
  /** For smoothing the probabilities in prior, posterior and probs */
  std::unique_ptr<polca_parallel::Smoother> smoother_;

 public:
  /**
   * Construct a new StandardError object
   *
   * Call Calc() and the resulting errors will be saved to prior_error and
   * prob_error
   *
   * @param features Not used
   * @param responses Design matrix of responses, matrix containing
   * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
   * has dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each category</li>
   * </ul>
   * @param probs Vector of probabilities for each outcome, for each category,
   * for each cluster flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param prior Design matrix of prior probabilities, probability data point
   * is in cluster m NOT given responses after calculations, it shall be in
   * matrix form with dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param posterior Design matrix of posterior probabilities (also called
   * responsibility), probability data point is in cluster m given responses
   * matrix
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_feature Number of features, required to be 1
   * @param n_outcomes Array of number of outcomes, for each category, and its
   * sum
   * @param n_cluster Number of clusters fitted
   * @param prior_error Vector to contain the standard error for the prior
   * probabilities for each cluster, modified after calling Calc()
   * @param prob_error Vector to contain the standard error for the outcome
   * probabilities category and cluster, modified after calling Calc()
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param regress_coeff_error Not used
   */
  StandardError(std::span<double> features, std::span<int> responses,
                std::span<double> probs, std::span<double> prior,
                std::span<double> posterior, std::size_t n_data,
                std::size_t n_feature, NOutcomes n_outcomes,
                std::size_t n_cluster, std::span<double> prior_error,
                std::span<double> prob_error,
                std::span<double> regress_coeff_error);

  /**
   * Calculate the standard errors
   *
   * Calculate the standard errors. Results are saved in the provided pointers
   * prior_error, prob_error and regress_coeff_error
   */
  void Calc();

 protected:
  /**
   * Smooth the probabilities prior, posterior and prob if a smoother exists
   *
   * Smooth the probabilities prior, posterior and prob if a smoother exists.
   * The pointers probs_, prior_ and posterior_ will point to the smoothed
   * probabilities.
   */
  void SmoothProbs();

  /** Instantiate and return an error_solver_*/
  virtual std::unique_ptr<polca_parallel::ErrorSolver> InitErrorSolver();

  /**
   * Calculate the scores
   *
   * Calculate the scores and saves it to the provided matrix
   *
   * @param score matrix to save the scores
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each parameter</li>
   * </ul>
   */
  void CalcScore(arma::Mat<double>& score);

  /**
   * Calculate the scores for the prior for all clusters except the zeroth one
   *
   * Calculate the scores for the prior for all clusters except the zeroth one
   * for all data points
   *
   * @param score_prior MODIFIED - submatrix of the complete score matrix, to
   * fill in with scores of the prior probabilities (and regression parameter if
   * applicable)
   */
  virtual void CalcScorePrior(arma::subview<double>& score_prior);

  /**
   * Calculate the scores for ALL outcome probabilities (except zeroth outcome)
   *
   * Calculate the scores for outcome probabilities for all clusters, categories
   * and outcomes (except for the zeroth outcome) for all data points.
   *
   * @param score_probs MODIFIED - submatrix of the complete score matrix, to
   * fill in with the scores of the outcome probabilities
   */
  void CalcScoreProbs(arma::subview<double>& score_probs);

  /**
   * Calculate the scores for the outcome probabilities
   *
   * Calculate the scores for the outcome probabilities for a given cluster,
   * category and outcome for all data points.
   *
   * @param outcome_index 1, 2, ..., n_outcomes[for a given category]
   * @param prob outcome probability for a given cluster, category and outcome
   * @param responses_j responses for a given category
   * @param posterior_i posterior for a given cluster
   * @param score_col MODIFIED, column of the score matrix to modify for a given
   * cluster, category and outcome
   */
  void CalcScoreProbsCol(std::size_t outcome_index, double prob,
                         arma::subview_col<int>& responses_j,
                         arma::subview_col<double>& posterior_i,
                         arma::subview_col<double>& score_col);

  /**
   * Calculate the Jacobian matrix
   *
   * Calculate the Jacobian matrix, a block diagonal matrix, and saves it in the
   * provided pointer
   *
   * @param jacobian pointer to save the Jacobian matrix
   */
  void CalcJacobian(arma::Mat<double>& jacobian);

  /**
   * Calculate the block matrix for the prior in the Jacobian matrix
   *
   * Calculate the block matrix for the prior in the Jacobian matrix and saves
   * it in the provided pointer. The provided pointer is shifted, ready for the
   * next block matrix.
   *
   * @param jacobian_ptr MODIFIED, the start of the block matrix in the Jacobian
   * matrix, the calculated block matrix is saved in *jacobian_ptr.
   * *jacobian_ptr is modified so that it points to the start of the next block
   * matrix after calling this method
   */
  virtual void CalcJacobianPrior(arma::subview<double>& jacobian_prior);

  /**
   * Calculate all block matrices for the probabilities in the Jacobian matrix
   *
   * Calculate all block matrices for the outcome probabilities in the Jacobian
   * matrix and saves it in the provided pointer. The provided pointer is
   * shifted, ready for the next block matrix.
   *
   * @param jacobian_ptr MODIFIED, the start of the block matrix in the Jacobian
   * matrix, the calculated block matrices are saved in *jacobian_ptr.
   * *jacobian_ptr is modified so that it points to the start of the next block
   * matrix after calling this method
   */
  void CalcJacobianProbs(arma::subview<double>& jacobian_probs);

  /**
   * Calculate a block matrix for given probabilities
   *
   * Calculate a block matrix for given probabilities and saves it in the
   * provided pointer. The provided pointer is shifted, ready for the next
   * block matrix.
   *
   * @param probs array of probabilities to construct the block matrix with
   * @param n_prob number of probabilities
   * @param jacobian_ptr MODIFIED, the start of the block matrix in the Jacobian
   * matrix, the calculated block matrix is saved in *jacobian_ptr.
   * *jacobian_ptr is modified so that it points to the start of the next block
   * matrix after calling this method
   */
  void CalcJacobianBlock(std::span<double> probs,
                         arma::subview<double>& jacobian_block);
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_STANDARD_ERROR_H
