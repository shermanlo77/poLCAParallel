// poLCAParallel
// Copyright (C) 2022 Sherman Lo

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

#include "em_algorithm.h"

const int polca_parallel::N_CATEGORY_SUMLOG = 100;

polca_parallel::EmAlgorithm::EmAlgorithm(
    double* features, int* responses, double* initial_prob, int n_data,
    int n_feature, int n_category, int* n_outcomes, int sum_outcomes,
    int n_cluster, int max_iter, double tolerance, double* posterior,
    double* prior, double* estimated_prob, double* regress_coeff) {
  this->features_ = features;
  this->responses_ = responses;
  this->initial_prob_ = initial_prob;
  this->n_data_ = n_data;
  this->n_feature_ = n_feature;
  this->n_category_ = n_category;
  this->n_outcomes_ = n_outcomes;
  this->sum_outcomes_ = sum_outcomes;
  this->n_cluster_ = n_cluster;
  this->max_iter_ = max_iter;
  this->tolerance_ = tolerance;
  this->posterior_ = posterior;
  this->prior_ = prior;
  this->estimated_prob_ = estimated_prob;
  this->regress_coeff_ = regress_coeff;
  this->ln_l_array_ = new double[this->n_data_];

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937_64* rng = new std::mt19937_64(seed);
  this->rng_ = std::unique_ptr<std::mt19937_64>(rng);
}

polca_parallel::EmAlgorithm::~EmAlgorithm() { delete[] this->ln_l_array_; }

void polca_parallel::EmAlgorithm::Fit() {
  bool is_first_run = true;
  bool is_success = false;

  double ln_l_difference;
  double ln_l_before;

  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  while (!is_success) {
    if (is_first_run) {
      // copy initial prob to estimated prob
      std::memcpy(this->estimated_prob_, this->initial_prob_,
                  this->n_cluster_ * this->sum_outcomes_ *
                      sizeof(*this->estimated_prob_));
    } else {
      // reach this condition if the first run has a problem
      // reset all required parameters
      this->Reset(&uniform);
    }

    // make a copy initial probabilities if requested
    if (this->best_initial_prob_ != NULL) {
      std::memcpy(this->best_initial_prob_, this->estimated_prob_,
                  this->n_cluster_ * this->sum_outcomes_ *
                      sizeof(*this->best_initial_prob_));
    }

    ln_l_before = -INFINITY;

    // initalise prior probabilities, for each cluster
    this->InitPrior();

    // do EM algorithm
    // assume successful until find error
    //
    // iteration goes:
    // initial E step (after which, n_iter=0, no effective iterations)
    // M step, E step (after which, n_iter=1, one effective iterations)
    // M step, E step (after which, n_iter=2, two effective iterations)
    // and so on until stopping condition
    // or when after an E step and n_iter=max_iter, making max_iter
    // effective iterations
    is_success = true;
    for (this->n_iter_ = 0; this->n_iter_ <= this->max_iter_; ++this->n_iter_) {
      // E step updates prior probabilities
      this->EStep();

      // E step updates ln_l_array_, use that to calculate log likelihood
      arma::Col<double> ln_l_array(this->ln_l_array_, this->n_data_, false);
      this->ln_l_ = sum(ln_l_array);

      // check for any errors
      ln_l_difference = this->ln_l_ - ln_l_before;
      if (this->IsInvalidLikelihood(ln_l_difference)) {
        is_success = false;
        break;
      }

      // check stopping conditions
      if (ln_l_difference < this->tolerance_) {
        break;
      }
      if (this->n_iter_ == this->max_iter_) {
        break;
      }
      ln_l_before = this->ln_l_;

      // M step updates posterior probabilities and estimated probabilities
      // break if m step has an error
      if (this->MStep()) {
        is_success = false;
        break;
      }
    }
    is_first_run = false;
  }

  // reformat prior
  this->FinalPrior();
}

// Set where to store initial probabilities (optional)
void polca_parallel::EmAlgorithm::set_best_initial_prob(
    double* best_initial_prob) {
  this->best_initial_prob_ = best_initial_prob;
}

double polca_parallel::EmAlgorithm::get_ln_l() { return this->ln_l_; }

int polca_parallel::EmAlgorithm::get_n_iter() { return this->n_iter_; }

bool polca_parallel::EmAlgorithm::get_has_restarted() {
  return this->has_restarted_;
}

void polca_parallel::EmAlgorithm::set_seed(unsigned seed) {
  std::mt19937_64* rng = new std::mt19937_64(seed);
  this->rng_ = std::unique_ptr<std::mt19937_64>(rng);
}

void polca_parallel::EmAlgorithm::set_rng(
    std::unique_ptr<std::mt19937_64>* rng) {
  this->rng_ = std::move(*rng);
}

std::unique_ptr<std::mt19937_64> polca_parallel::EmAlgorithm::move_rng() {
  return std::move(this->rng_);
}

void polca_parallel::EmAlgorithm::Reset(
    std::uniform_real_distribution<double>* uniform) {
  // generate random number for estimated_prob_
  this->has_restarted_ = true;
  polca_parallel::GenerateNewProb(this->rng_.get(), uniform, this->n_outcomes_,
                                  this->sum_outcomes_, this->n_category_,
                                  this->n_cluster_, this->estimated_prob_);
}

void polca_parallel::EmAlgorithm::InitPrior() {
  // prior probabilities are the same for all data points in this
  // implementation
  for (int i = 0; i < this->n_cluster_; ++i) {
    this->prior_[i] = 1.0 / static_cast<double>(this->n_cluster_);
  }
}

void polca_parallel::EmAlgorithm::FinalPrior() {
  // Copying prior probabilities as each data point as the same prior
  double prior_copy[this->n_cluster_];
  std::memcpy(&prior_copy, this->prior_,
              this->n_cluster_ * sizeof(*this->prior_));
  for (int m = 0; m < this->n_cluster_; ++m) {
    for (int i = 0; i < this->n_data_; ++i) {
      this->prior_[m * this->n_data_ + i] = prior_copy[m];
    }
  }
}

double polca_parallel::EmAlgorithm::GetPrior(int data_index,
                                             int cluster_index) {
  return this->prior_[cluster_index];
}

void polca_parallel::EmAlgorithm::EStep() {
  double* estimated_prob;  // for pointing to elements in estimated_prob_
  // used for likelihood calculation for a data point
  // P(Y^{(i)})
  double normaliser;

  for (int i = 0; i < this->n_data_; ++i) {
    // normaliser noramlise over cluster, so loop over cluster here
    normaliser = 0.0;

    estimated_prob = this->estimated_prob_;
    for (int m = 0; m < this->n_cluster_; ++m) {
      this->PosteriorUnnormalize(i, m, &estimated_prob);
      normaliser += this->posterior_[m * this->n_data_ + i];
    }
    // normalise
    for (int m = 0; m < this->n_cluster_; ++m) {
      this->posterior_[m * this->n_data_ + i] /= normaliser;
    }
    // store the log likelihood for this data point
    this->ln_l_array_[i] = log(normaliser);
  }
}

void polca_parallel::EmAlgorithm::PosteriorUnnormalize(
    int data_index, int cluster_index, double** estimated_prob) {
  int* responses_i = this->responses_ + (data_index * this->n_category_);
  double prior = this->GetPrior(data_index, cluster_index);
  double posterior = polca_parallel::PosteriorUnnormalize(
      responses_i, this->n_category_, this->n_outcomes_, estimated_prob, prior);
  this->posterior_[cluster_index * this->n_data_ + data_index] = posterior;
}

bool polca_parallel::EmAlgorithm::IsInvalidLikelihood(double ln_l_difference) {
  return isnan(this->ln_l_);
}

bool polca_parallel::EmAlgorithm::MStep() {
  // estimate prior
  // for this implementation, the mean posterior, taking the mean over data
  // points
  arma::Mat<double> posterior_arma(this->posterior_, this->n_data_,
                                   this->n_cluster_, false);
  arma::Row<double> prior = mean(posterior_arma, 0);
  std::memcpy(this->prior_, prior.begin(),
              this->n_cluster_ * sizeof(*this->prior_));

  // estimate outcome probabilities
  this->EstimateProbability();

  return false;
}

void polca_parallel::EmAlgorithm::EstimateProbability() {
  // set all estimated response probability to zero
  for (int i = 0; i < this->n_cluster_ * this->sum_outcomes_; ++i) {
    this->estimated_prob_[i] = 0.0;
  }

  // for each cluster
  for (int m = 0; m < this->n_cluster_; ++m) {
    // estimate outcome probabilities
    this->WeightedSumProb(m);
    this->NormalWeightedSumProb(m);
  }
}

void polca_parallel::EmAlgorithm::WeightedSumProb(int cluster_index) {
  int n_outcome;
  int y;
  double posterior_iter;
  // point to outcome probabilites for given cluster
  double* estimated_prob_m =
      this->estimated_prob_ + cluster_index * this->sum_outcomes_;
  double* estimated_prob;
  for (int i = 0; i < this->n_data_; ++i) {
    estimated_prob = estimated_prob_m;
    for (int j = 0; j < this->n_category_; ++j) {
      n_outcome = this->n_outcomes_[j];
      y = this->responses_[i * this->n_category_ + j];
      posterior_iter = this->posterior_[cluster_index * this->n_data_ + i];
      estimated_prob[y - 1] += posterior_iter;
      // point to next category
      estimated_prob += n_outcome;
    }
  }
}

void polca_parallel::EmAlgorithm::NormalWeightedSumProb(int cluster_index) {
  this->NormalWeightedSumProb(
      cluster_index,
      static_cast<double>(this->n_data_) * this->prior_[cluster_index]);
}

void polca_parallel::EmAlgorithm::NormalWeightedSumProb(int cluster_index,
                                                        double normaliser) {
  int n_outcome;
  // point to outcome probabilites for given cluster
  double* estimated_prob =
      this->estimated_prob_ + cluster_index * this->sum_outcomes_;
  // normalise by the sum of posteriors
  // calculations can be reused as the prior is the mean of posteriors
  // from the E step
  for (int j = 0; j < this->n_category_; ++j) {
    n_outcome = this->n_outcomes_[j];
    for (int k = 0; k < n_outcome; ++k) {
      estimated_prob[k] /= normaliser;
    }
    estimated_prob += n_outcome;
  }
}

double polca_parallel::PosteriorUnnormalize(
    int* responses_i, int n_category,
    int* n_outcomes, double** estimated_prob, double prior) {

  // used for calculating posterior probability, conditioned on a cluster m,
  // for a data point
  // P(cluster m | Y^{(i)})
  double posterior;
  int y;  // for getting a response from responses_

  if (n_category < polca_parallel::N_CATEGORY_SUMLOG) {
    // used for conditioned on cluster m likelihood calculation
    // for a data point
    // P(Y^{(i)} | cluster m)
    double p = 1;

    // calculate conditioned on cluster m likelihood
    for (int j = 0; j < n_category; ++j) {
      y = responses_i[j];
      p *= (*estimated_prob)[y - 1];
      // increment to point to the next category
      *estimated_prob += n_outcomes[j];
    }
    // posterior = likelihood x prior
    posterior = p * prior;
  } else {

    double ln_p = 0;
    // calculate conditioned on cluster m likelihood
    for (int j = 0; j < n_category; ++j) {
      y = responses_i[j];
      ln_p += log((*estimated_prob)[y - 1]);
      // increment to point to the next category
      *estimated_prob += n_outcomes[j];
    }
    posterior = ln_p + log(prior);
    posterior = exp(posterior);
  }

  return posterior;
}

void polca_parallel::GenerateNewProb(
    std::mt19937_64* rng, std::uniform_real_distribution<double>* uniform,
    int* n_outcomes, int sum_outcomes, int n_category, int n_cluster,
    double* prob) {
  for (double* ptr = prob; ptr < prob + n_cluster * sum_outcomes; ++ptr) {
    *ptr = (*uniform)(*rng);
  }
  // normalise to probabilities
  int n_outcome;
  for (int m = 0; m < n_cluster; ++m) {
    for (int j = 0; j < n_category; ++j) {
      n_outcome = n_outcomes[j];
      arma::Col<double> prob_vector(prob, n_outcome, false);
      prob_vector /= sum(prob_vector);
      prob += n_outcome;
    }
  }
}
