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

#include "RcppArmadillo.h"

polca_parallel::Smoother::Smoother(int* responses, double* probs, double* prior,
                                   double* posterior, int n_data,
                                   int n_category, int* n_outcomes,
                                   int sum_outcomes, int n_cluster)
    : responses_(responses),
      probs_(probs, probs + sum_outcomes * n_cluster),
      prior_(prior, prior + n_cluster * n_data),
      posterior_(posterior, posterior + n_cluster * n_data),
      n_data_(n_data),
      n_category_(n_category),
      n_outcomes_(n_outcomes),
      sum_outcomes_(sum_outcomes),
      n_cluster_(n_cluster) {
  // std::vector makes a copy of the array
}

void polca_parallel::Smoother::Smooth() {
  // use posterior to get the estimate of number of data in each cluster
  arma::Mat<double> posterior(this->posterior_.data(), this->n_data_,
                              this->n_cluster_, false);
  arma::Row<double> n_data = arma::sum(posterior, 0);

  // smooth outcome probabilities
  double* probs = this->probs_.data();
  for (int i_cluster = 0; i_cluster < this->n_cluster_; ++i_cluster) {
    for (int* n_outcome = this->n_outcomes_;
         n_outcome < this->n_outcomes_ + this->n_category_; ++n_outcome) {
      this->Smooth(probs, *n_outcome, n_data[i_cluster], 1.0,
                   static_cast<double>(*n_outcome));
      probs += *n_outcome;
    }
  }

  // perhaps smooth prior as well
  // for posterior update, use E step in EmAlgorithm
}

double* polca_parallel::Smoother::get_probs() { return this->probs_.data(); }

double* polca_parallel::Smoother::get_prior() { return this->prior_.data(); }

double* polca_parallel::Smoother::get_posterior() {
  return this->posterior_.data();
}

void polca_parallel::Smoother::Smooth(double* probs, int length, double n_data,
                                      double num_add, double deno_add) {
  arma::Col<double> probs_arma(probs, length, false, true);
  probs_arma = (n_data * probs_arma + num_add) / (n_data + deno_add);
}
