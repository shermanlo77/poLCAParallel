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

#include "standard_error.h"

polca_parallel::StandardError::StandardError(
    double* features, int* responses, double* probs, double* prior,
    double* posterior, int n_data, int n_feature, int n_category,
    int* n_outcomes, int sum_outcomes, int n_cluster, double* prior_error,
    double* prob_error, double* regress_coeff_error) {
  this->features_ = features;
  this->responses_ = responses;
  this->probs_ = probs;
  this->prior_ = prior;
  this->posterior_ = posterior;
  this->n_data_ = n_data;
  this->n_feature_ = n_feature;
  this->n_category_ = n_category;
  this->n_outcomes_ = n_outcomes;
  this->sum_outcomes_ = sum_outcomes;
  this->n_cluster_ = n_cluster;
  this->prior_error_ = prior_error;
  this->prob_error_ = prob_error;
  this->regress_coeff_error_ = regress_coeff_error;
}

void polca_parallel::StandardError::Calc() {
  // calculate the inverse information matrix
  arma::Mat<double> info_arma = this->CalcInfo();
  arma::Mat<double> info_arma_inv = arma::pinv(info_arma);

  // calculate the Jacobian matrix
  int height = this->GetInfoSize();
  int width = this->GetJacobianWidth();
  double* jacobian = new double[height * width];
  this->CalcJacobian(jacobian);

  // calculate the covariance matrix
  arma::Mat<double> jacobian_arma(jacobian, height, width, false);
  arma::Mat<double> jacobian_arma_t(jacobian, height, width, true);
  arma::inplace_trans(jacobian_arma_t);

  arma::Mat<double> covariance =
      jacobian_arma_t * info_arma_inv * jacobian_arma;

  delete[] jacobian;

  // extract prior error from the covariance matrix
  double* covariance_ptr = covariance.begin();
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    this->prior_error_[cluster_index] = sqrt(*covariance_ptr);
    covariance_ptr += width + 1;
  }

  // extract probs error from the covariance matrix
  for (int prob_index = 0; prob_index < this->sum_outcomes_ * this->n_cluster_;
       ++prob_index) {
    this->prob_error_[prob_index] = sqrt(*covariance_ptr);
    covariance_ptr += width + 1;
  }
}

arma::Mat<double> polca_parallel::StandardError::CalcInfo() {
  int info_size = this->GetInfoSize();
  double* score = new double[this->n_data_ * info_size];
  this->CalcScore(score);
  arma::Mat<double> score_arma(score, this->n_data_, info_size, false);
  arma::Mat<double> score_arma_t(score, this->n_data_, info_size, true);
  arma::inplace_trans(score_arma_t);
  arma::Mat<double> info_arma = score_arma_t * score_arma;
  delete[] score;
  return info_arma;
}

int polca_parallel::StandardError::GetInfoSize() {
  int info_size = (this->n_cluster_ - 1) +
                  this->n_cluster_ * (this->sum_outcomes_ - this->n_category_);
  return info_size;
}

int polca_parallel::StandardError::GetJacobianWidth() {
  int jacobian_width =
      this->n_cluster_ + this->n_cluster_ * this->sum_outcomes_;
  return jacobian_width;
}

void polca_parallel::StandardError::CalcScore(double* score) {
  // for cache efficient, iterate for each data point in inner loop
  //
  // each call of CalcScorePrior and CalcScoreProbs fills in n_data elements of
  // the score design matrix, so shift the score pointer by n_data after every
  // call of those methods

  for (int cluster_index = 1; cluster_index < this->n_cluster_;
       ++cluster_index) {
    this->CalcScorePrior(cluster_index, score);
    score += this->n_data_;
  }

  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    for (int category_index = 0; category_index < this->n_category_;
         ++category_index) {
      for (int outcome_index = 1;
           outcome_index < this->n_outcomes_[category_index]; ++outcome_index) {
        this->CalcScoreProbs(outcome_index, category_index, cluster_index,
                             score);
        score += this->n_data_;
      }
    }
  }
}

void polca_parallel::StandardError::CalcScorePrior(int cluster_index,
                                                   double* score_start) {
  double* prior = this->prior_ + cluster_index * this->n_data_;
  double* posterior = this->posterior_ + cluster_index * this->n_data_;
  // iterate for each data point
  for (double* score = score_start; score < score_start + this->n_data_;
       ++score) {
    *score = *posterior - *prior;
    ++prior;
    ++posterior;
  }
}

void polca_parallel::StandardError::CalcScoreProbs(int outcome_index,
                                                   int category_index,
                                                   int cluster_index,
                                                   double* score_start) {
  double* posterior = this->posterior_ + cluster_index * this->n_data_;
  int* response = this->responses_ + category_index * this->n_data_;
  double* prob = this->probs_ + cluster_index * this->sum_outcomes_;
  for (int i = 0; i < category_index; ++i) {
    prob += this->n_outcomes_[i];
  }
  prob += outcome_index;

  // boolean if the response is the same as the outcome
  // remember response is one index, not zero index
  int is_outcome;
  // iterate for each data point
  for (double* score = score_start; score < score_start + this->n_data_;
       ++score) {
    is_outcome = outcome_index == (*response - 1);
    *score = *posterior * (((double)is_outcome) - *prob);
    ++posterior;
    ++response;
  }
}

void polca_parallel::StandardError::CalcJacobian(double* jacobian) {
  int height = this->GetInfoSize();
  int width = this->GetJacobianWidth();

  // create a block diagonal matrix
  std::fill(jacobian, jacobian + height * width, 0.0);

  // block matrix for prior
  this->CalcJacobianPrior(&jacobian, height);

  // block matrix for probs, one for each cluster and category pair
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    for (int category_index = 0; category_index < this->n_category_;
         ++category_index) {
      this->CalcJacobianProbs(category_index, cluster_index, &jacobian, height);
    }
  }
}

void polca_parallel::StandardError::CalcJacobianPrior(double** jacobian_ptr,
                                                      int info_size) {
  // copy over the prior, they will be the same for all data points
  double prior[this->n_cluster_];
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    prior[cluster_index] = this->prior_[cluster_index * this->n_data_];
  }

  // fill in the block matrix
  double jac_element;
  double* jacobian = *jacobian_ptr;
  for (int j_cluster = 0; j_cluster < this->n_cluster_; ++j_cluster) {
    for (int i_cluster = 1; i_cluster < this->n_cluster_; ++i_cluster) {
      jac_element = -prior[i_cluster] * prior[j_cluster];
      if (i_cluster == j_cluster) {
        jac_element += prior[i_cluster];
      }
      *jacobian = jac_element;
      ++jacobian;
    }
    *jacobian_ptr += info_size;
    jacobian = *jacobian_ptr;
  }

  // shift, ready for the next block matrix
  *jacobian_ptr += this->n_cluster_ - 1;
}

void polca_parallel::StandardError::CalcJacobianProbs(int category_index,
                                                      int cluster_index,
                                                      double** jacobian_ptr,
                                                      int info_size) {
  // work out the probability for the given category and cluster
  double* prob = this->probs_ + cluster_index * this->sum_outcomes_;
  for (int i = 0; i < category_index; ++i) {
    prob += this->n_outcomes_[i];
  }

  int n_outcome = this->n_outcomes_[category_index];

  // fill in the block matrix
  double jac_element;
  double* jacobian = *jacobian_ptr;
  for (int j_outcome = 0; j_outcome < n_outcome; ++j_outcome) {
    for (int i_outcome = 1; i_outcome < n_outcome; ++i_outcome) {
      jac_element = -prob[i_outcome] * prob[j_outcome];
      if (i_outcome == j_outcome) {
        jac_element += prob[i_outcome];
      }
      *jacobian = jac_element;
      ++jacobian;
    }
    *jacobian_ptr += info_size;
    jacobian = *jacobian_ptr;
  }

  // shift, ready for the next block matrix
  *jacobian_ptr += n_outcome - 1;
}
