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

polca_parallel::StandardErrorRegress::StandardErrorRegress(
    double* features, int* responses, double* probs, double* prior,
    double* posterior, int n_data, int n_feature, int n_category,
    int* n_outcomes, int sum_outcomes, int n_cluster, double* prior_error,
    double* prob_error, double* regress_coeff_error)
    : polca_parallel::StandardError(
          features, responses, probs, prior, posterior, n_data, n_feature,
          n_category, n_outcomes, sum_outcomes, n_cluster, prior_error,
          prob_error, regress_coeff_error) {}

void polca_parallel::StandardErrorRegress::CalcScorePrior(double** score) {
  arma::Mat<double> features_arma(this->features_, this->n_data_,
                                  this->n_feature_, false);
  for (int cluster_index = 1; cluster_index < this->n_cluster_;
       ++cluster_index) {
    arma::Mat<double> score_i_arma(*score, this->n_data_, this->n_feature_,
                                   false, true);
    arma::Col<double> prior_i_arma(this->prior_ + cluster_index * this->n_data_,
                                   this->n_data_, false);
    arma::Col<double> posterior_i_arma(
        this->posterior_ + cluster_index * this->n_data_, this->n_data_, false);

    score_i_arma = features_arma.each_col() % (posterior_i_arma - prior_i_arma);
    *score += this->n_data_ * this->n_feature_;
  }
}

void polca_parallel::StandardErrorRegress::CalcJacobianPrior(
    double** jacobian_ptr) {
  double jac_element;
  double* jacobian = *jacobian_ptr;
  double prior_i;
  double prior_j;
  double* feature;
  for (int j_cluster = 0; j_cluster < this->n_cluster_; ++j_cluster) {
    for (int i_cluster = 1; i_cluster < this->n_cluster_; ++i_cluster) {
      feature = this->features_;
      for (int i_feature = 0; i_feature < this->n_feature_; ++i_feature) {
        for (int i_data = 0; i_data < this->n_data_; ++i_data) {
          prior_i = this->prior_[i_cluster * this->n_data_ + i_data];
          prior_j = this->prior_[j_cluster * this->n_data_ + i_data];
          jac_element = -prior_i * prior_j;
          if (i_cluster == j_cluster) {
            jac_element += prior_i;
          }
          *jacobian += jac_element * *feature++;
        }
        *jacobian /= (double)this->n_data_;
        ++jacobian;
      }
    }
    *jacobian_ptr += this->info_size_;
    jacobian = *jacobian_ptr;
  }
  // shift, ready for the next block matrix
  *jacobian_ptr += (this->n_cluster_ - 1) * this->n_feature_;
}

void polca_parallel::StandardErrorRegress::ExtractErrorGivenInfoInv(
    double* info_inv, double* jacobian) {
  int size = this->n_feature_ * (this->n_cluster_ - 1);
  arma::Mat<double> info_arma(info_inv, this->info_size_, this->info_size_,
                              false);
  arma::Mat<double> regress_coeff_error(this->regress_coeff_error_, size, size,
                                        false, true);
  regress_coeff_error = info_arma.submat(0, 0, size - 1, size - 1);
  this->polca_parallel::StandardError::ExtractErrorGivenInfoInv(info_inv,
                                                                jacobian);
}
