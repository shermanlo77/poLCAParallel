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

polca_parallel::Smoother::Smoother(double* probs, double* prior,
                                   double* posterior, int n_data,
                                   int n_category, int* n_outcomes,
                                   int sum_outcomes, int n_cluster)
    : probs_(probs, probs + sum_outcomes * n_cluster),
      prior_(prior, prior + n_cluster * n_data),
      posterior_(posterior, posterior + n_cluster * n_data),
      n_data_(n_data),
      n_category_(n_category),
      n_outcomes_(n_outcomes),
      sum_outcomes_(sum_outcomes),
      n_cluster_(n_cluster) {}

void polca_parallel::Smoother::Smooth() {
  double* probs = this->probs_.data();
  int n_outcome;
  for (int i_cluster = 0; i_cluster < this->n_cluster_; ++i_cluster) {
    for (int* n_outcome = this->n_outcomes_;
         n_outcome < this->n_outcomes_ + this->n_category_; ++n_outcome) {
      this->Smooth(probs, *n_outcome, this->n_data_, 1.0, *n_outcome);
      probs += *n_outcome;
    }
  }
  this->Smooth(this->prior_.data(), this->prior_.size(), this->n_data_, 1.0,
               this->n_cluster_);
  this->Smooth(this->posterior_.data(), this->posterior_.size(), this->n_data_,
               1.0, this->n_cluster_);
}

double* polca_parallel::Smoother::get_probs() { return this->probs_.data(); }

double* polca_parallel::Smoother::get_prior() { return this->prior_.data(); }

double* polca_parallel::Smoother::get_posterior() {
  return this->posterior_.data();
}

void polca_parallel::Smoother::Smooth(double* probs, int length, int n_data,
                                      double num_add, int deno_add) {
  arma::Col<double> probs_arma(probs, length, false, true);
  double n_data_double = static_cast<double>(n_data);
  double deno = static_cast<double>(n_data_double + deno_add);
  probs_arma = (n_data_double * probs_arma + num_add) / deno;
}
