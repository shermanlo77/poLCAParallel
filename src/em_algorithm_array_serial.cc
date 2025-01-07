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

#include "em_algorithm_array_serial.h"

#include <memory>
#include <random>
#include <utility>

polca_parallel::EmAlgorithmArraySerial::EmAlgorithmArraySerial(
    double* features, int* responses, double* initial_prob, std::size_t n_data,
    std::size_t n_feature, std::size_t n_category, std::size_t* n_outcomes,
    std::size_t sum_outcomes, std::size_t n_cluster, std::size_t n_rep,
    unsigned int max_iter, double tolerance, double* posterior, double* prior,
    double* estimated_prob, double* regress_coeff)
    : polca_parallel::EmAlgorithmArray(
          features, responses, initial_prob, n_data, n_feature, n_category,
          n_outcomes, sum_outcomes, n_cluster, n_rep, 1, max_iter, tolerance,
          posterior, prior, estimated_prob, regress_coeff) {}

void polca_parallel::EmAlgorithmArraySerial::SetSeed(std::seed_seq* seed) {
  this->seed_array_ = std::make_unique<unsigned[]>(1);
  unsigned* seed_array = this->seed_array_.get();
  seed->generate(seed_array, seed_array + 1);
  this->rng_ = std::make_unique<std::mt19937_64>(seed_array[0]);
}

void polca_parallel::EmAlgorithmArraySerial::SetSeed(unsigned seed) {
  this->seed_array_ = std::make_unique<unsigned[]>(1);
  this->seed_array_.get()[0] = seed;
  this->rng_ = std::make_unique<std::mt19937_64>(seed);
}

void polca_parallel::EmAlgorithmArraySerial::SetRng(
    std::unique_ptr<std::mt19937_64>* rng) {
  this->rng_ = std::move(*rng);
}

std::unique_ptr<std::mt19937_64>
polca_parallel::EmAlgorithmArraySerial::MoveRng() {
  return std::move(this->rng_);
}

void polca_parallel::EmAlgorithmArraySerial::SetFitterRng(
    polca_parallel::EmAlgorithm* fitter, std::size_t rep_index) {
  if (this->rng_) {
    fitter->set_rng(&this->rng_);
  }
}

void polca_parallel::EmAlgorithmArraySerial::MoveRngBackFromFitter(
    polca_parallel::EmAlgorithm* fitter) {
  // do not check this->rng != NULL as it will always be NULL after calling
  // SetFitterRng()
  //
  // if the rng was set before hand, it will be set to null after transferring
  // ownership to a EmAlgorithm object, thus this method does need to be called
  // to set rng back to what it was
  //
  // A EmAlgorithm object will always have a rng even if no rng has been set
  //
  // So if no rng has been set, at the end of a Fit(), rng will be set to the
  // default instantiated rng
  this->rng_ = fitter->move_rng();
}
