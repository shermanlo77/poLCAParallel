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
#include "util.h"

template class polca_parallel::EmAlgorithmNanTemplate<
    polca_parallel::EmAlgorithm>;

template class polca_parallel::EmAlgorithmNanTemplate<
    polca_parallel::EmAlgorithmRegress>;

template <typename T>
polca_parallel::EmAlgorithmNanTemplate<T>::EmAlgorithmNanTemplate(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, unsigned int max_iter, double tolerance,
    std::span<double> posterior, std::span<double> prior,
    std::span<double> estimated_prob, std::span<double> regress_coeff)
    : T(features, responses, initial_prob, n_data, n_feature, n_outcomes,
        n_cluster, max_iter, tolerance, posterior, prior, estimated_prob,
        regress_coeff),
      posterior_sum_(n_outcomes.size()) {}

template <typename T>
void polca_parallel::EmAlgorithmNanTemplate<T>::WeightedSumProb(
    const std::size_t cluster_index) {
  polca_parallel::NanWeightedSumProb(
      cluster_index, this->responses_, this->n_outcomes_, this->posterior_,
      this->estimated_prob_, this->posterior_sum_);
}

template <typename T>
void polca_parallel::EmAlgorithmNanTemplate<T>::NormalWeightedSumProb(
    const std::size_t cluster_index) {
  polca_parallel::NanNormalWeightedSumProb(cluster_index, this->n_outcomes_,
                                           this->posterior_sum_,
                                           this->estimated_prob_);
}

template <typename T>
double polca_parallel::EmAlgorithmNanTemplate<T>::PosteriorUnnormalize(
    std::span<const int> responses_i, double prior,
    const arma::Col<double>& estimated_prob) const {
  return polca_parallel::PosteriorUnnormalize<true>(
      responses_i, this->n_outcomes_, estimated_prob, prior);
}

polca_parallel::EmAlgorithmNan::EmAlgorithmNan(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, unsigned int max_iter, double tolerance,
    std::span<double> posterior, std::span<double> prior,
    std::span<double> estimated_prob, std::span<double> regress_coeff)
    : EmAlgorithmNanTemplate<EmAlgorithm>(
          features, responses, initial_prob, n_data, n_feature, n_outcomes,
          n_cluster, max_iter, tolerance, posterior, prior, estimated_prob,
          regress_coeff) {}

polca_parallel::EmAlgorithmNanRegress::EmAlgorithmNanRegress(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, unsigned int max_iter, double tolerance,
    std::span<double> posterior, std::span<double> prior,
    std::span<double> estimated_prob, std::span<double> regress_coeff)
    : EmAlgorithmNanTemplate<EmAlgorithmRegress>(
          features, responses, initial_prob, n_data, n_feature, n_outcomes,
          n_cluster, max_iter, tolerance, posterior, prior, estimated_prob,
          regress_coeff) {}

void polca_parallel::NanWeightedSumProb(const std::size_t cluster_index,
                                        std::span<const int> responses,
                                        std::span<const std::size_t> n_outcomes,
                                        const arma::Mat<double>& posterior,
                                        arma::Mat<double>& estimated_prob,
                                        std::vector<double>& posterior_sum) {
  std::fill(posterior_sum.begin(), posterior_sum.end(), 0.0);

  auto y = responses.begin();
  // point to outcome probabilites for given cluster for the zeroth category
  arma::Col<double> estimated_prob_col =
      estimated_prob.unsafe_col(cluster_index);
  arma::Col<double>::iterator estimated_prob_iter;

  for (double posterior_i : posterior.unsafe_col(cluster_index)) {
    estimated_prob_iter = estimated_prob_col.begin();
    std::size_t i_category = 0;
    for (std::size_t n_outcome_j : n_outcomes) {
      // selective summing of posterior
      if (*y > 0) {
        *std::next(estimated_prob_iter, *y - 1) += posterior_i;
        posterior_sum[i_category] += posterior_i;
      }
      // point to next category
      std::advance(y, 1);
      std::advance(estimated_prob_iter, n_outcome_j);
      ++i_category;
    }
  }
}

void polca_parallel::NanNormalWeightedSumProb(
    const std::size_t cluster_index, std::span<const std::size_t> n_outcomes,
    std::vector<double>& posterior_sum, arma::Mat<double>& estimated_prob) {
  auto estimated_prob_col = estimated_prob.unsafe_col(cluster_index).begin();
  for (std::size_t i_category = 0; i_category < n_outcomes.size();
       ++i_category) {
    std::size_t n_outcome = n_outcomes[i_category];
    arma::Col<double> estimated_prob_i(estimated_prob_col, n_outcome, false,
                                       true);
    estimated_prob_i /= posterior_sum[i_category];
    std::advance(estimated_prob_col, n_outcome);
  }
}
