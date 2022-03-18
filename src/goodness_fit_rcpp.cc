#include <array>
#include <map>

#include "RcppArmadillo.h"

#include "goodness_fit.h"

// GOODNESS OF FIT STATISTICS
// Get goodness of fit statistics given fitted probabilities
// Args:
  // features: design matrix of features
  // prior: vector of prior probabilities, for each cluster
  // outcome_prob: vector of response probabilities for each cluster, flatten
    // list of matrices, from the return value of poLCA.vectorize.R
    // flatten list of matrices
      // dim 0: for each outcome
      // dim 1: for each category
      // dim 2: for each cluster
  // n_data: number of data points
  // n_category: number of categories
  // n_outcomes: vector, number of possible responses for each category
  // n_cluster: number of clusters, or classes, to fit
// Return a list:
  // unique_freq_table: data frame of unique responses with their observed
      // frequency and expected frequency
  // ln_l_ratio
  // chi_squared
// [[Rcpp::export]]
Rcpp::List GoodnessFitRcpp(
    Rcpp::IntegerMatrix responses,
    Rcpp::NumericVector prior,
    Rcpp::NumericVector outcome_prob,
    int n_data,
    int n_category,
    Rcpp::IntegerVector n_outcomes,
    int n_cluster) {

  // get observed and expected frequencies for each unique response
  std::map<std::vector<int>, polca_parallel::Frequency>* unique_freq =
      new std::map<std::vector<int>, polca_parallel::Frequency>();
  GetUniqueObserved(responses.begin(), n_data, n_category, unique_freq);
  GetExpected(responses.begin(),
              prior.begin(),
              outcome_prob.begin(),
              n_data,
              n_category,
              n_outcomes.begin(),
              n_cluster,
              unique_freq);
  // get log likelihood ratio and chi squared statistics
  std::array<double, 2> stats = GetStatistics(unique_freq, n_data);

  // transfer results from std::map unique_freq to a NumericMatrix
      // unique_freq_table
  // last two columns for observed and expected frequency
  int n_unique = unique_freq->size();
  Rcpp::NumericMatrix unique_freq_table(n_unique, n_category+2);
  double* unique_freq_ptr = unique_freq_table.begin();
  std::vector<int> response_i;

  int data_index = 0;
  polca_parallel::Frequency frequency;
  double expected, observed;
  for (auto iter=unique_freq->begin();
      iter!=unique_freq->end(); iter++) {

    response_i = iter->first;
    frequency = iter->second;
    expected = frequency.expected;
    observed = (double) frequency.observed;

    // copy over response
    for (int j=0; j<n_category; j++) {
      unique_freq_ptr[j*n_unique+data_index] = response_i[j];
    }
    // copy over observed and expected frequency
    unique_freq_ptr[n_category*n_unique+data_index] = observed;
    unique_freq_ptr[(n_category+1)*n_unique+data_index] = expected;
    data_index++;
  }

  Rcpp::List to_return;
  to_return.push_back(unique_freq_table);
  to_return.push_back(stats[0]);
  to_return.push_back(stats[1]);

  delete unique_freq;

  return to_return;
}
