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

#include "em_algorithm_nan.h"

#include <algorithm>
#include <vector>

#include "RcppArmadillo.h"
#include "em_algorithm.h"
#include "em_algorithm_regress.h"

polca_parallel::EmAlgorithmNan::EmAlgorithmNan(
    double* features, int* responses, double* initial_prob, int n_data,
    int n_feature, int n_category, int* n_outcomes, int sum_outcomes,
    int n_cluster, int max_iter, double tolerance, double* posterior,
    double* prior, double* estimated_prob, double* regress_coeff)
    : polca_parallel::EmAlgorithm(
          features, responses, initial_prob, n_data, n_feature, n_category,
          n_outcomes, sum_outcomes, n_cluster, max_iter, tolerance, posterior,
          prior, estimated_prob, regress_coeff),
      posterior_sum_(n_category) {}

void polca_parallel::EmAlgorithmNan::WeightedSumProb(int cluster_index) {
  polca_parallel::NanWeightedSumProb(
      cluster_index, this->responses_, this->n_data_, this->n_category_,
      this->n_outcomes_, this->sum_outcomes_, this->posterior_,
      this->estimated_prob_, &(this->posterior_sum_));
}

void polca_parallel::EmAlgorithmNan::NormalWeightedSumProb(int cluster_index) {
  polca_parallel::NanNormalWeightedSumProb(
      cluster_index, this->n_category_, this->n_outcomes_, this->sum_outcomes_,
      &(this->posterior_sum_), this->estimated_prob_);
}

polca_parallel::EmAlgorithmNanRegress::EmAlgorithmNanRegress(
    double* features, int* responses, double* initial_prob, int n_data,
    int n_feature, int n_category, int* n_outcomes, int sum_outcomes,
    int n_cluster, int max_iter, double tolerance, double* posterior,
    double* prior, double* estimated_prob, double* regress_coeff)
    : polca_parallel::EmAlgorithmRegress(
          features, responses, initial_prob, n_data, n_feature, n_category,
          n_outcomes, sum_outcomes, n_cluster, max_iter, tolerance, posterior,
          prior, estimated_prob, regress_coeff),
      posterior_sum_(n_category) {}

void polca_parallel::EmAlgorithmNanRegress::WeightedSumProb(int cluster_index) {
  polca_parallel::NanWeightedSumProb(
      cluster_index, this->responses_, this->n_data_, this->n_category_,
      this->n_outcomes_, this->sum_outcomes_, this->posterior_,
      this->estimated_prob_, &(this->posterior_sum_));
}

void polca_parallel::EmAlgorithmNanRegress::NormalWeightedSumProb(
    int cluster_index) {
  polca_parallel::NanNormalWeightedSumProb(
      cluster_index, this->n_category_, this->n_outcomes_, this->sum_outcomes_,
      &(this->posterior_sum_), this->estimated_prob_);
}

void polca_parallel::NanWeightedSumProb(int cluster_index, int* responses,
                                        int n_data, int n_category,
                                        int* n_outcomes, int sum_outcomes,
                                        double* posterior,
                                        double* estimated_prob,
                                        std::vector<double>* posterior_sum) {
  std::fill(posterior_sum->begin(), posterior_sum->end(), 0.0);

  int y;
  double posterior_i;
  // point to outcome probabilites for given cluster for the zeroth category
  double* estimated_prob_start = estimated_prob + cluster_index * sum_outcomes;
  double* estimated_prob_i;  // pointer to prob for i_category
  for (int i_data = 0; i_data < n_data; ++i_data) {
    estimated_prob_i = estimated_prob_start;
    for (int i_category = 0; i_category < n_category; ++i_category) {
      // selective summing of posterior
      y = responses[i_data * n_category + i_category];
      if (y > 0) {
        posterior_i = posterior[cluster_index * n_data + i_data];
        estimated_prob_i[y - 1] += posterior_i;
        posterior_sum->at(i_category) += posterior_i;
      }
      // point to next category
      estimated_prob_i += n_outcomes[i_category];
    }
  }
}

void polca_parallel::NanNormalWeightedSumProb(
    int cluster_index, int n_category, int* n_outcomes, int sum_outcomes,
    std::vector<double>* posterior_sum, double* estimated_prob) {
  double* estimated_prob_ptr = estimated_prob + cluster_index * sum_outcomes;
  int n_outcome;
  for (int i_category = 0; i_category < n_category; ++i_category) {
    n_outcome = n_outcomes[i_category];
    arma::Col<double> estimated_prob(estimated_prob_ptr, n_outcome, false,
                                     true);
    estimated_prob /= posterior_sum->at(i_category);
    estimated_prob_ptr += n_outcome;
  }
}
