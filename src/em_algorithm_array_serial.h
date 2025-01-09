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

#ifndef POLCAPARALLEL_SRC_EM_ALGORITHM_ARRAY_SERIAL_H_
#define POLCAPARALLEL_SRC_EM_ALGORITHM_ARRAY_SERIAL_H_

#include <memory>
#include <random>

#include "em_algorithm_array.h"

namespace polca_parallel {

/**
 * Serial version of EmAlgorithmArray
 *
 * Only uses one thread (so the parameter n_thread is not provided) and each
 * repetition reuses one rng, rather than each repetition having a rng each.
 * Thus the member variable seed_array_ shall only contain one seed. The rng is
 * only used for creating new initial values should a repetition fail.
 *
 * This is used by Blrt where each thread works on different bootstrap samples
 * in parallel. This ensures no additional threads are spawned.
 *
 */
class EmAlgorithmArraySerial : public polca_parallel::EmAlgorithmArray {
 private:
  /** The one and only random number generator to be used by all repetitions */
  std::unique_ptr<std::mt19937_64> rng_;

 public:
  /**
   * @copydoc EmAlgorithmArraySerial::EmAlgorithmArray
   * @param n_thread omitted EmAlgorithmArraySerial
   */
  EmAlgorithmArraySerial(double* features, int* responses, double* initial_prob,
                         std::size_t n_data, std::size_t n_feature,
                         std::size_t n_category, std::size_t* n_outcomes,
                         std::size_t sum_outcomes, std::size_t n_cluster,
                         std::size_t n_rep, unsigned int max_iter,
                         double tolerance, double* posterior, double* prior,
                         double* estimated_prob, double* regress_coeff);

  /**
   * Set the seed_array_ to contain only one seed and instantiate the rng_
   *
   * The rng is only used if a repetition fails and tries again using new
   * initial values generated by the rng.
   */
  void SetSeed(std::seed_seq& seed) override;

  /**
   * Set the seed_array_ to contain only one seed and instantiate the rng_
   *
   * The rng is only used if a repetition fails and tries again using new
   * initial values generated by the rng.
   */
  void SetSeed(unsigned seed);

  /**
   * Transfer ownership of a rng to this object and set rng_
   *
   * Transfer ownership of a rng to this object and set rng_. This rng is only
   * used if a repetition fails and tries again using new initial values
   * generated by the rng.
   */
  void SetRng(std::unique_ptr<std::mt19937_64>* rng);

  /**
   * Transfer ownership of the rng from this object as a return value
   */
  std::unique_ptr<std::mt19937_64> MoveRng();

 protected:
  /**
   * Set the rng of an EmAlgorithm object
   *
   * This will transfer ownership of rng_ from this object to the fitter. Ensure
   * to call MoveRngBackFromFitter() to retrieve it back afterwards.
   *
   * Because each repetition reuses the same rng, the parameter rep_index is
   * ignored.
   */
  void SetFitterRng(polca_parallel::EmAlgorithm& fitter,
                    std::size_t rep_index) override;

  /**
   * Transfer ownership of an EmAlgorithm's rng back to this object's rng_
   */
  void MoveRngBackFromFitter(polca_parallel::EmAlgorithm& fitter) override;
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_EM_ALGORITHM_ARRAY_SERIAL_H_
