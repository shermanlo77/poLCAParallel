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

#ifndef POLCAPARALLEL_SRC_STANDARD_ERROR_REGRESS_H_
#define POLCAPARALLEL_SRC_STANDARD_ERROR_REGRESS_H_

#include <memory>
#include <span>

#include "standard_error.h"
#include "util.h"

namespace polca_parallel {

/**
 * For calculating the standard errors of the fitted poLCA regression parameters
 *
 * See the superclass StandardError. This implementation caters for poLCA
 * regression models
 */
class StandardErrorRegress : public polca_parallel::StandardError {
 protected:
  /**
   * Design matrix of features, matrix with dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each feature</li>
   * </ul>
   */
  arma::Mat<double> features_;

 public:
  /**
   * Construct a new StandardErrorRegress object
   *
   * Call Calc() and the resulting errors will be saved to prior_error,
   * prob_error and regress_coeff_error
   *
   * @param features Design matrix of features, matrix with dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each feature</li>
   * </ul>
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
   * @param n_feature Number of features
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
   * @param regress_coeff_error Matrix to contain the covariance matrix of the
   * regression coefficient, modified after calling Calc()
   */
  StandardErrorRegress(std::span<double> features, std::span<int> responses,
                       std::span<double> probs, std::span<double> prior,
                       std::span<double> posterior, std::size_t n_data,
                       std::size_t n_feature, NOutcomes n_outcomes,
                       std::size_t n_cluster, std::span<double> prior_error,
                       std::span<double> prob_error,
                       std::span<double> regress_coeff_error);

 protected:
  [[nodiscard]] std::unique_ptr<polca_parallel::ErrorSolver> InitErrorSolver()
      override;
  void CalcScorePrior(arma::subview<double>& score_prior) const override;
  void CalcJacobianPrior(arma::subview<double>& jacobian_prior) const override;
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_STANDARD_ERROR_REGRESS_H_
