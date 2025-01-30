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

#include "regularised_error.h"

#include <memory>
#include <span>

#include "RcppArmadillo.h"
#include "smoother.h"
#include "standard_error.h"
#include "standard_error_regress.h"
#include "util.h"

polca_parallel::RegularisedError::RegularisedError(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> probs, std::span<double> prior,
    std::span<double> posterior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : polca_parallel::StandardError(
          features, responses, probs, prior, posterior, n_data, n_feature,
          n_outcomes, n_cluster, prior_error, prob_error, regress_coeff_error) {
  this->smoother_ = std::make_unique<polca_parallel::Smoother>(
      this->probs_, this->prior_, this->posterior_, this->n_data_,
      this->n_outcomes_, this->n_cluster_);
}

polca_parallel::RegularisedRegressError::RegularisedRegressError(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> probs, std::span<double> prior,
    std::span<double> posterior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    std::span<double> prior_error, std::span<double> prob_error,
    std::span<double> regress_coeff_error)
    : polca_parallel::StandardErrorRegress(
          features, responses, probs, prior, posterior, n_data, n_feature,
          n_outcomes, n_cluster, prior_error, prob_error, regress_coeff_error) {
  this->smoother_ = std::make_unique<polca_parallel::Smoother>(
      this->probs_, this->prior_, this->posterior_, this->n_data_,
      this->n_outcomes_, this->n_cluster_);
}
