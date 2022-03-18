#ifndef GOODNESS_FIT_H_
#define GOODNESS_FIT_H_

#include <array>
#include <map>
#include <math.h>
#include <stdexcept>

#include "RcppArmadillo.h"

namespace polca_parallel {

struct Frequency {
  int observed;
  double expected;
};

// GET UNIQUE OBSERVED
// Iterate through all the responses, then find and count unique combinations of
    // outcomes which were observed in the dataset. Results are stored in a map.
// Parameters:
  // responses: design matrix transpose of responses, matrix n_category x n_data
  // n_data: number of data points
  // n_category: number of categories
  // unique_freq: map to write results to, key: unique integer vectors,
      // value: Frequency containing the number of times it was observed in the
          // dataset
void GetUniqueObserved (
    int* responses,
    int n_data,
    int n_category,
    std::map<std::vector<int>, Frequency>* unique_freq) {

  // iterate through each data point
  std::vector<int> response_i(n_category);
  for (int i=0; i<n_data; i++) {
    for (int j=0; j<n_category; j++) {
      response_i[j] = responses[j];
    }
    // add or update observation count
    try {
      unique_freq->at(response_i).observed++;
    } catch (std::out_of_range e) {
      Frequency frequency;
      frequency.observed = 1;
      unique_freq->insert({response_i, frequency});
    }
    responses += n_category;
  }
}

// GET EXPECTED FREQUENCY
// Update the expected frequency in a map of vector<int>:Frequency by modifying
    // the value of Frequency.expected with the likelihood of that unique
    // reponse with multiplied by n_data
// Parameters:
  // responses: design matrix transpose of responses, matrix n_category x n_data
  // prior: vector of prior probabilities (probability in a cluster),
      // length n_cluster
  // outcome_prob: array of outcome probabilities for each category and
      // cluster,
      // flatten list of matrices
        // dim 0: for each outcome
        // dim 1: for each category
        // dim 2: for each model
  // n_data: number of data points
  // n_category: number of categories
  // n_outcomes: array of integers, number of outcomes for each category, array
      // of length n_category
  // n_cluster: number of clusters (or classes)
  // unique_freq: map to modify with results to, key: unique integer vectors,
      // value: Frequency with the expected frequency modified
void GetExpected (
    int* responses,
    double* prior,
    double* outcome_prob,
    int n_data,
    int n_category,
    int* n_outcomes,
    int n_cluster,
    std::map<std::vector<int>, Frequency>* unique_freq) {

  double p;
  double total_p;
  double* outcome_prob_ptr;
  int n_outcome;
  int y;
  std::vector<int> response_i;

  // iterate through the map
  for (auto iter=unique_freq->begin();
      iter!=unique_freq->end(); iter++) {

    // calculate likelihood
    response_i = iter->first;

    total_p = 0.0;  // to be summed over all clusters
    outcome_prob_ptr = outcome_prob;
    // iterate through each cluster
    for (int m=0; m<n_cluster; m++) {
      // calculate likelihood conditioned on cluster m
      p = 1;
      for (int j=0; j<n_category; j++) {
        n_outcome = n_outcomes[j];
        y = response_i[j];
        p *= outcome_prob_ptr[y - 1];
        // increment to point to the next category
        outcome_prob_ptr += n_outcome;
      }
      p *= prior[m];
      total_p += p;
    }

    iter->second.expected = total_p * n_data;
  }
}

// GET CHI-SQUARED and LOG LIKELIHOOD RATIO STATISTICS
// Calculate and return the chi-squared statistics and log likelihood ratio
// Parameters:
  // unique_freq: map of unique responses and their frequencies, both observed
      // and expected
  // n_data: number of data points
// Return:
  // log likelihood ratio
  // chi squared statistics
std::array<double, 2> GetStatistics(
    std::map<std::vector<int>, Frequency>* unique_freq,
    int n_data) {

  std::vector<int> response_i;
  Frequency frequency;

  int n_unique = unique_freq->size();

  // store statistics for each unique response
  double* chi_squared_array = new double[n_unique];
  double* ln_l_ratio_array = new double[n_unique];
  double* expected_array = new double[n_unique];
  double expected;
  double observed;
  double diff_squared;

  // extract and calculate statistics for each one each unique response,
  int index = 0;
  for (auto iter=unique_freq->begin();
      iter!=unique_freq->end(); iter++) {
    frequency = iter->second;
    expected = frequency.expected;
    observed = (double) frequency.observed;

    diff_squared = (expected-observed);
    diff_squared *= diff_squared;

    expected_array[index] = expected;
    chi_squared_array[index] = diff_squared / expected;
    ln_l_ratio_array[index] = observed * log(observed / expected);
    index++;
  }

  double chi_squared;
  double ln_l_ratio;

  // chi squared calculation also use unobserved responses
  chi_squared = arma::sum(arma::Row<double>(chi_squared_array, n_unique, false))
      + ((double) n_data
         - arma::sum(arma::Row<double>(expected_array, n_unique, false)));
  ln_l_ratio = 2.0 * sum(arma::Row<double>(ln_l_ratio_array, n_unique, false));

  delete[] chi_squared_array;
  delete[] ln_l_ratio_array;
  delete[] expected_array;

  return {ln_l_ratio, chi_squared};
}

}

#endif
