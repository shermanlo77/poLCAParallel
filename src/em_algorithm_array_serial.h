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

#ifndef EM_ALGORITHM_ARRAY_SERIAL_H_
#define EM_ALGORITHM_ARRAY_SERIAL_H_

#include "em_algorithm_array.h"

namespace polca_parallel {

/**
 * Serial version of EmAlgorithmArray. Only uses one thread (so the parameter
 * n_thread is ignored) and each repetition reuse one rng, rather than each
 * repetition having a rng each. The rng is only used for creating new initial
 * values should a repetition fail.
 */
class EmAlgorithmArraySerial : public polca_parallel::EmAlgorithmArray {
 private:
  /** The one and only random number generator */
  std::unique_ptr<std::mt19937_64> rng_;

 public:
  /** @copydoc EmAlgorithmArraySerial::EmAlgorithmArray
   */
  EmAlgorithmArraySerial(double* features, int* responses, double* initial_prob,
                         int n_data, int n_feature, int n_category,
                         int* n_outcomes, int sum_outcomes, int n_cluster,
                         int n_rep, int n_thread, int max_iter,
                         double tolerance, double* posterior, double* prior,
                         double* estimated_prob, double* regress_coeff,
                         bool is_regress);

  /**
   * Set the member variable seed_array_ containing only one seed and the rng.
   * Rng only used if a repetition fails and tries again using new initial
   * values.
   */
  void SetSeed(std::seed_seq* seed) override;

  /**
   * Set the member variable seed_array_ containing only one seed and the rng.
   * Rng only used if a repetition fails and tries again using new initial
   * values.
   */
  void SetSeed(unsigned seed);

  /**
   * Transfer ownership of a rng to here and set rng_. Rng only used if a
   * repetition fails and tries again using new initial values.
   */
  void SetRng(std::unique_ptr<std::mt19937_64>* rng);

  /**
   * Transfer ownership of the rng from here as a return value
   */
  std::unique_ptr<std::mt19937_64> MoveRng();

 protected:
  /**
   * Set the rng of a EmAlgorithm object. Because each repetition reuses the
   * same rng, the parameter rep_index is ignored.
   */
  void SetFitterRng(polca_parallel::EmAlgorithm* fitter,
                    int rep_index) override;

  void MoveRngBackFromFitter(polca_parallel::EmAlgorithm* fitter) override;
};

}  // namespace polca_parallel

#endif  // EM_ALGORITHM_ARRAY_SERIAL_H_
