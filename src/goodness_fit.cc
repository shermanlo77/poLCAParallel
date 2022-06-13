#include "goodness_fit.h"

void polca_parallel::GetUniqueObserved(
    int* responses, int n_data, int n_category,
    std::map<std::vector<int>, Frequency>* unique_freq) {
  // iterate through each data point
  std::vector<int> response_i(n_category);
  for (int i = 0; i < n_data; ++i) {
    for (int j = 0; j < n_category; ++j) {
      response_i[j] = responses[j];
    }
    // add or update observation count
    try {
      ++unique_freq->at(response_i).observed;
    } catch (std::out_of_range e) {
      Frequency frequency;
      frequency.observed = 1;
      unique_freq->insert({response_i, frequency});
    }
    responses += n_category;
  }
}

void polca_parallel::GetExpected(
    int* responses, double* prior, double* outcome_prob, int n_data,
    int n_category, int* n_outcomes, int n_cluster,
    std::map<std::vector<int>, Frequency>* unique_freq) {
  double p;
  double total_p;
  double* outcome_prob_ptr;
  int n_outcome;
  int y;
  std::vector<int> response_i;

  // iterate through the map
  for (auto iter = unique_freq->begin(); iter != unique_freq->end(); ++iter) {
    // calculate likelihood
    response_i = iter->first;

    total_p = 0.0;  // to be summed over all clusters
    outcome_prob_ptr = outcome_prob;
    // iterate through each cluster
    for (int m = 0; m < n_cluster; ++m) {
      // calculate likelihood conditioned on cluster m
      p = 1;
      for (int j = 0; j < n_category; ++j) {
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

std::array<double, 2> polca_parallel::GetStatistics(
    std::map<std::vector<int>, Frequency>* unique_freq, int n_data) {
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
  for (auto iter = unique_freq->begin(); iter != unique_freq->end(); ++iter) {
    frequency = iter->second;
    expected = frequency.expected;
    observed = static_cast<double>(frequency.observed);

    diff_squared = (expected - observed);
    diff_squared *= diff_squared;

    expected_array[index] = expected;
    chi_squared_array[index] = diff_squared / expected;
    ln_l_ratio_array[index] = observed * log(observed / expected);
    ++index;
  }

  double chi_squared;
  double ln_l_ratio;

  // chi squared calculation also use unobserved responses
  chi_squared =
      arma::sum(arma::Row<double>(chi_squared_array, n_unique, false)) +
      (static_cast<double>(n_data) -
       arma::sum(arma::Row<double>(expected_array, n_unique, false)));
  ln_l_ratio = 2.0 * sum(arma::Row<double>(ln_l_ratio_array, n_unique, false));

  delete[] chi_squared_array;
  delete[] ln_l_ratio_array;
  delete[] expected_array;

  return {ln_l_ratio, chi_squared};
}
