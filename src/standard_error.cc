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
  this->info_size_ =
      this->n_feature_ * (this->n_cluster_ - 1) +
      this->n_cluster_ * (this->sum_outcomes_ - this->n_category_);
  this->jacobian_width_ =
      this->n_cluster_ + this->n_cluster_ * this->sum_outcomes_;
}

void polca_parallel::StandardError::Calc() {
  // calcaulte the info matrix
  double* info = new double[this->info_size_ * this->info_size_];
  this->CalcInfo(info);

  // calculate the Jacobian matrix
  double* jacobian = new double[this->info_size_ * this->jacobian_width_];
  this->CalcJacobian(jacobian);

  this->ExtractError(info, jacobian);

  delete[] jacobian;
  delete[] info;
}

void polca_parallel::StandardError::CalcInfo(double* info) {
  arma::Mat<double> info_arma(info, this->info_size_, this->info_size_, false,
                              true);
  double* score = new double[this->n_data_ * this->info_size_];
  this->CalcScore(score);
  arma::Mat<double> score_arma(score, this->n_data_, this->info_size_, false);
  arma::Mat<double> score_arma_t(score, this->n_data_, this->info_size_, true);
  arma::inplace_trans(score_arma_t);
  info_arma = score_arma_t * score_arma;
  delete[] score;
}

void polca_parallel::StandardError::CalcScore(double* score) {
  // for cache efficient, iterate for each data point in inner loop
  //
  // each call of CalcScorePrior and CalcScoreProbs fills in coumns of the score
  // design matrix
  // score is shifted automatically by these methods
  this->CalcScorePrior(&score);
  this->CalcScoreProbs(&score);
}

void polca_parallel::StandardError::CalcScorePrior(double** score) {
  // call CalcScorePriorCol() for every cluster except the zeroth one
  for (int cluster_index = 1; cluster_index < this->n_cluster_;
       ++cluster_index) {
    this->CalcScorePriorCol(cluster_index, *score);
    *score += this->n_data_;
  }
}

void polca_parallel::StandardError::CalcScorePriorCol(int cluster_index,
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

void polca_parallel::StandardError::CalcScoreProbs(double** score) {
  // call CalcScoreProbsCol() for every cluster, category and outcome except
  // for the zeroth outcome
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    for (int category_index = 0; category_index < this->n_category_;
         ++category_index) {
      for (int outcome_index = 1;
           outcome_index < this->n_outcomes_[category_index]; ++outcome_index) {
        this->CalcScoreProbsCol(outcome_index, category_index, cluster_index,
                                *score);
        *score += this->n_data_;
      }
    }
  }
}

void polca_parallel::StandardError::CalcScoreProbsCol(int outcome_index,
                                                      int category_index,
                                                      int cluster_index,
                                                      double* score_start) {
  // start posterior for the given cluster
  double* posterior = this->posterior_ + cluster_index * this->n_data_;
  // start the response for the given category
  int* response = this->responses_ + category_index * this->n_data_;
  // start the prob for the given cluster, category and outcome triplet
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
  // create a block diagonal matrix
  std::fill(jacobian, jacobian + this->info_size_ * this->jacobian_width_, 0.0);

  // block matrix for prior
  this->CalcJacobianPrior(&jacobian);
  this->CalcJacobianProbs(&jacobian);
}

void polca_parallel::StandardError::CalcJacobianPrior(double** jacobian_ptr) {
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
    *jacobian_ptr += this->info_size_;
    jacobian = *jacobian_ptr;
  }

  // shift, ready for the next block matrix
  *jacobian_ptr += this->n_cluster_ - 1;
}

void polca_parallel::StandardError::CalcJacobianProbs(double** jacobian_ptr) {
  // block matrix for probs, one for each cluster and category pair
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    for (int category_index = 0; category_index < this->n_category_;
         ++category_index) {
      this->CalcJacobianProbsBlock(category_index, cluster_index, jacobian_ptr);
    }
  }
}

void polca_parallel::StandardError::CalcJacobianProbsBlock(
    int category_index, int cluster_index, double** jacobian_ptr) {
  // work out the probability for the given category and cluster pair
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
    *jacobian_ptr += this->info_size_;
    jacobian = *jacobian_ptr;
  }

  // shift, ready for the next block matrix
  *jacobian_ptr += n_outcome - 1;
}

void polca_parallel::StandardError::ExtractError(double* info,
                                                 double* jacobian) {
  // calculate the inverse information matrix
  arma::Mat<double> info_arma(info, this->info_size_, this->info_size_, false);
  arma::Mat<double> info_arma_inv = arma::pinv(info_arma);

  this->ExtractErrorGiveInfoInv(info_arma_inv.memptr(), jacobian);
}

void polca_parallel::StandardError::ExtractErrorGiveInfoInv(double* info_inv,
                                                            double* jacobian) {
  arma::Mat<double> info_arma_inv(info_inv, this->info_size_, this->info_size_,
                                  false);

  arma::Mat<double> jacobian_arma(jacobian, this->info_size_,
                                  this->jacobian_width_, false);
  arma::Mat<double> jacobian_arma_t(jacobian, this->info_size_,
                                    this->jacobian_width_, true);
  arma::inplace_trans(jacobian_arma_t);

  // calculate the covariance matrix
  arma::Mat<double> covariance_arma =
      jacobian_arma_t * info_arma_inv * jacobian_arma;

  double* covariance = covariance_arma.memptr();
  // extract prior error from the covariance matrix
  this->ExtractErrorPrior(&covariance);
  // extract probs error from the covariance matrix
  this->ExtractErrorProb(&covariance);
}

void polca_parallel::StandardError::ExtractErrorPrior(double** covariance) {
  // extract diagonal elements
  int width = this->jacobian_width_;
  for (int cluster_index = 0; cluster_index < this->n_cluster_;
       ++cluster_index) {
    this->prior_error_[cluster_index] = sqrt(**covariance);
    *covariance += width + 1;
  }
}

void polca_parallel::StandardError::ExtractErrorProb(double** covariance) {
  // extract diagonal elements
  int width = this->jacobian_width_;
  for (int prob_index = 0; prob_index < this->sum_outcomes_ * this->n_cluster_;
       ++prob_index) {
    this->prob_error_[prob_index] = sqrt(**covariance);
    *covariance += width + 1;
  }
}
