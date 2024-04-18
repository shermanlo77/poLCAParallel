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

#ifndef STANDARD_ERROR_REGRESS_H
#define STANDARD_ERROR_REGRESS_H

#include "standard_error.h"

namespace polca_parallel {

/**
 * For calculating the standard errors of the fitted poLCA regression parameters
 *
 * See the superclass StandardError. This implementation caters for poLCA
 * regression models
 */
class StandardErrorRegress : public polca_parallel::StandardError {
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
   * @param n_category Number of categories
   * @param n_outcomes Array of number of outcomes, for each category
   * @param sum_outcomes Sum of all integers in n_outcomes
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
  StandardErrorRegress(double* features, int* responses, double* probs,
                       double* prior, double* posterior, int n_data,
                       int n_feature, int n_category, int* n_outcomes,
                       int sum_outcomes, int n_cluster, double* prior_error,
                       double* prob_error, double* regress_coeff_error);

 protected:
  void CalcScorePrior(double** score_start) override;
  void CalcJacobianPrior(double** jacobian_ptr) override;
  void ExtractErrorGivenInfoInv(double* info_inv, double* jacobian) override;
};

}  // namespace polca_parallel

#endif  // STANDARD_ERROR_REGRESS_H
