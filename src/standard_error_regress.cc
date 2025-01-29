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

#include "standard_error_regress.h"

#include <memory>
#include <span>

#include "RcppArmadillo.h"
#include "error_solver.h"
#include "standard_error.h"
#include "util.h"

polca_parallel::StandardErrorRegress::StandardErrorRegress(
    std::span<double> features, std::span<int> responses,
    std::span<double> probs, std::span<double> prior,
    std::span<double> posterior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : polca_parallel::StandardError(
          features, responses, probs, prior, posterior, n_data, n_feature,
          n_outcomes, n_cluster, prior_error, prob_error, regress_coeff_error),
      features_(features.data(), n_data, n_feature, false, true) {}

std::unique_ptr<polca_parallel::ErrorSolver>
polca_parallel::StandardErrorRegress::InitErrorSolver() {
  return std::make_unique<polca_parallel::InfoEigenRegressSolver>(
      this->n_data_, this->n_feature_, this->n_outcomes_.sum(),
      this->n_cluster_, this->info_size_, this->jacobian_width_,
      this->prior_error_, this->prob_error_, this->regress_coeff_error_);
}

void polca_parallel::StandardErrorRegress::CalcScorePrior(
    arma::subview<double>& score_prior) {
  for (std::size_t cluster_index = 1; cluster_index < this->n_cluster_;
       ++cluster_index) {
    auto score_prior_i =
        score_prior.cols((cluster_index - 1) * this->n_feature_,
                         cluster_index * this->n_feature_ - 1);
    score_prior_i = this->features_.each_col() %
                    (this->posterior_.unsafe_col(cluster_index) -
                     this->prior_.unsafe_col(cluster_index));
  }
}

void polca_parallel::StandardErrorRegress::CalcJacobianPrior(
    arma::subview<double>& jacobian_prior) {
  auto jacobian = jacobian_prior.begin();
  for (std::size_t j_cluster = 0; j_cluster < this->n_cluster_; ++j_cluster) {
    for (std::size_t i_cluster = 1; i_cluster < this->n_cluster_; ++i_cluster) {
      auto feature = this->features_.begin();
      for (std::size_t i_feature = 0; i_feature < this->n_feature_;
           ++i_feature) {
        for (std::size_t i_data = 0; i_data < this->n_data_; ++i_data) {
          double prior_i = this->prior_[i_cluster * this->n_data_ + i_data];
          double prior_j = this->prior_[j_cluster * this->n_data_ + i_data];
          double jac_element = -prior_i * prior_j;
          if (i_cluster == j_cluster) {
            jac_element += prior_i;
          }
          *jacobian += jac_element * *feature;

          std::advance(feature, 1);
        }
        *jacobian /= static_cast<double>(this->n_data_);
        std::advance(jacobian, 1);
      }
    }
  }
}
