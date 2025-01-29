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

#include "smoother.h"

#include <span>

#include "RcppArmadillo.h"
#include "util.h"

polca_parallel::Smoother::Smoother(std::span<double> probs,
                                   std::span<double> prior,
                                   std::span<double> posterior,
                                   std::size_t n_data, NOutcomes n_outcomes,
                                   std::size_t n_cluster)
    : probs_(probs.begin(), probs.end()),
      prior_(prior.begin(), prior.end()),
      posterior_(posterior.begin(), posterior.end()),
      n_data_(n_data),
      n_outcomes_(n_outcomes),
      n_cluster_(n_cluster) {
  // std::vector makes a copy of the array
}

void polca_parallel::Smoother::Smooth() {
  // use posterior to get the estimate of number of data in each cluster
  arma::Mat<double> posterior(this->posterior_.data(), this->n_data_,
                              this->n_cluster_, false, true);
  arma::Row<double> n_data = arma::sum(posterior, 0);

  // smooth outcome probabilities
  double* probs = this->probs_.data();
  for (std::size_t i_cluster = 0; i_cluster < this->n_cluster_; ++i_cluster) {
    for (std::size_t n_outcome_j : this->n_outcomes_) {
      this->Smooth(probs, n_outcome_j, n_data[i_cluster], 1.0,
                   static_cast<double>(n_outcome_j));
      probs += n_outcome_j;
    }
  }

  // perhaps smooth prior as well
  // for posterior update, use E step in EmAlgorithm
}

std::span<double> polca_parallel::Smoother::get_probs() {
  return std::span<double>(this->probs_.begin(), this->probs_.size());
}

std::span<double> polca_parallel::Smoother::get_prior() {
  return std::span<double>(this->prior_.begin(), this->prior_.size());
}

std::span<double> polca_parallel::Smoother::get_posterior() {
  return std::span<double>(this->posterior_.begin(), this->posterior_.size());
}

void polca_parallel::Smoother::Smooth(double* probs, std::size_t length,
                                      double n_data, double num_add,
                                      double deno_add) {
  arma::Col<double> probs_arma(probs, length, false, true);
  probs_arma = (n_data * probs_arma + num_add) / (n_data + deno_add);
}
