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

#ifndef POLCAPARALLEL_SRC_SMOOTHER_H
#define POLCAPARALLEL_SRC_SMOOTHER_H

#include <vector>

#include "RcppArmadillo.h"
#include "em_algorithm.h"

namespace polca_parallel {

/**
 * For smoothing the probabilities probs, prior and posterior in StandardError
 *
 * For smoothing the probabilities probs, prior and posterior in StandardError.
 * It creates a copy of these probabilities and smoothes it using Laplace
 * smoothing (or Bayesian). It can be interpreted as adjusting your probability
 * estimated as if there were additional data. This prevents probabilities from
 * being exactly 0.0 or 1.0 by adding a bias to the estimate.
 *
 * How to use
 * <ul>
 *   <li>
 *     Pass the probabilities and other information to the constructor. They
 *     will be member variables of StandardError.
 *   </li>
 *   <li>Call the method Smooth()</li>
 *   <li>
 *     The pointer to the smoothed probabilities are available via the methods
 *     get_probs(), get_prior() and get_posterior()
 *   </li>
 * </ul>
 */
class Smoother {
 private:
  int* responses_;
  /**
   * Vector of smoothed probabilities for the outcome probabilities. Flatten
   * list in the order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * */
  std::vector<double> probs_;
  /**
   * Design matrix of smoothed prior probabilities.  Matrix with the following
   * dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  std::vector<double> prior_;
  /**
   * Design matrix of smoothed posterior probabilities.  Matrix with the
   * following dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  std::vector<double> posterior_;
  /** Number of data points */
  int n_data_;
  /** Number of categories */
  int n_category_;
  /** Vector of the number of outcomes for each category */
  int* n_outcomes_;
  /** Sum of n_outcomes */
  int sum_outcomes_;
  /** Number of clusters to fit */
  int n_cluster_;

 public:
  /**
   * Construct a new Smoother object
   *
   * Creates a copy of probs, prior and posterior. Call Smooth() to smooth
   * these probabilities
   *
   * @param responses Design matrix of responses, matrix containing
   * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
   * has dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each category</li>
   * </ul>
   * @param probs Vector of probabilities for each outcome, for each category,
   * for each cluster flatten list in the order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param prior Design matrix of prior probabilities, matrix form with
   * dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param posterior Design matrix of posterior probabilities, matrix form with
   * dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_category Number of categories
   * @param n_outcomes Array of number of outcomes, for each category
   * @param sum_outcomes Sum of all integers in n_outcomes
   * @param n_cluster Number of clusters
   */
  Smoother(int* responses, double* probs, double* prior, double* posterior,
           int n_data, int n_category, int* n_outcomes, int sum_outcomes,
           int n_cluster);

  /**
   * Smooth the probabilities probs_, prior_ and posterior_
   *
   * Smooth the probabilities probs_, prior_ and posterior_. This method
   * replaces the content of the vectors.
   */
  void Smooth();

  /** Get the pointer to the array of smoothed probs */
  double* get_probs();

  /** Get the pointer to the array of prior probs */
  double* get_prior();

  /** Get the pointer to the array of posterior probs */
  double* get_posterior();

 private:
  /**
   * Do Laplace smoothing given probabilities
   *
   * (n_data * probs + num_add) / (n_data + demo_add)
   *
   * @param probs array of probabilities to modify
   * @param length length of the array probs
   * @param n_data number of data points used to estimate the probabilities
   * @param num_add see equation
   * @param demo_add see equation
   */
  void Smooth(double* probs, int length, double n_data, double num_add,
              double demo_add);
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_SMOOTHER_H
