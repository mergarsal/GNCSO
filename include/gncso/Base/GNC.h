#pragma once
#include <algorithm>
#include <cmath>
#include <experimental/optional>
#include <functional>
#include <iostream>
#include <limits>

#include "gncso/Base/BaseConcepts.h"
#include "gncso/Base/Concepts.h"
#include "gncso/GM/GM_fcn.h"
#include "gncso/TLS/TLS_fcn.h"

// for the general fcns
using namespace baseOpt;

namespace gncso {

template <typename Variable, typename Weights, typename Scalar=double, typename... Args>
GNCResult<Variable, Weights, Scalar>
GNC( const VariableEstimation<Variable, Weights, Args...>& estimate_model_fcn,
     const ComputeCost<Variable, Weights, Scalar, Args...>& compute_cost_fcn,
     const ComputeResiduals<Variable, Weights, Args...>& compute_residuals_fcn,
     const InitializationMu<Weights, Scalar, Args...>& initialize_mu_fcn,
     const UpdateWeights<Variable, Scalar, Weights, Args...>& update_weights_fcn,
     const UpdateMu<Scalar, Args...>& update_mu_fcn,
     const Variable& initial_estimation_x,
     const Weights& initial_weights,
     Args &... args,
     const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
     const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
     const GNCParams<Scalar> & params = GNCParams<Scalar>())
     {
       std::cout << "---------------Inside GNC function---------------\n";

    // Output struct
    GNCResult<Variable, Weights, Scalar> result_gnc;
    result_gnc.GNCstatus = GNCStatus::ITERATION_LIMIT;

    // Current iterate and proposed next iterates;
    Variable x, x_proposed;

    // Function value at the current iterate and proposed iterate
    Scalar f_x, f_x_proposed, f_outer_prev;
    f_x = 10000000;  // Initial value
    f_outer_prev = f_x;

    Scalar mu, mu_proposed;

    // Weights at the current iterate
    Weights weights, weights_proposed;

    // Residuals at the current iterate
    Weights residuals;

    // Define var. for inliers
    size_t n = weights.size();
    Weights set_inliers;
    set_inliers.setZero(1, n);


    // Initialize Variable
    x = initial_estimation_x;
    // Initialize weights
    weights = initial_weights;

    //  Update parameters
    if (update_params_in_fcn)
      {
        (*update_params_in_fcn)(initial_estimation_x, initial_weights, args...);
      }

    // Initialize mu
    residuals = compute_residuals_fcn(initial_estimation_x, args...);

    // initialize mu
    mu = initialize_mu_fcn(residuals, params, args...);

    // Setting up initial weights
    // update_weights_in_fcn(x, weights, args...);

    size_t i_outer_iter = 0;  // to make it visible outside the loop
    size_t i_inner_iter = 0;  // to make it visible outside the loop
    Scalar diff_costs;

    // Check if mu < 0
    // if so, Break
    if (fabs(mu) < params.mu_threshold)
    {
      if (params.GNC_verbose)           std::cout << "Initial mu was close to 0. \n";
      result_gnc.GNCstatus = GNCStatus::MU_ZERO;
    }
    else
    {
      // run outer iteration
      std::cout << "###############\nStarting GNC\n###############\n";

      for (i_outer_iter = 0; i_outer_iter < params.max_outer_iterations; i_outer_iter++)
      {
        GNCOuterResult<Variable, Weights, Scalar> outer_results;
        // Start of outer loop
        if (params.log_outer_iters)       outer_results.GNCouterstatus = GNCOuterStatus::MAX_NR_ITERS;

        // Run inner loop
        for (i_inner_iter = 0; i_inner_iter < params.max_inner_iterations; i_inner_iter++)
        {
            std::cout << "Outer iter: " << i_outer_iter << "               mu: " << mu;
            std::cout << "                    Inner iter: " << i_inner_iter << std::endl;

            // Note that here the parameters are already saved!
            // 1.0 Initialize variable if required
            if (initialize_variable_fcn)
            {
              x = (*initialize_variable_fcn)(x, weights, args...);
            }
            // 1. Fix weights and estimate E
            x_proposed = estimate_model_fcn(x, weights, args...);

            // 2. Compute residuals
            residuals = compute_residuals_fcn(x_proposed, args...);

            // 3. Fix X and compute weights
            weights_proposed = update_weights_fcn(x_proposed, weights, residuals, mu, params, args...);

            // 3.1 Check if we have enough data
            set_inliers.setZero();  // clear
            for (size_t i = 0; i < n; i++)    set_inliers(i) = (weights_proposed(i) > params.inliers_threshold);
            // compute nr. of inliers
            if (set_inliers.sum() < params.nr_min_points)
            {
                // we have less points than required!!
                // OPtion 1: reset weights
                weights_proposed = initial_weights;
            }

            // 3.2. Update params
            if (update_params_in_fcn)
            {
              (*update_params_in_fcn)(x_proposed, weights_proposed, args...);
            }
            // 4. Compute cost
            f_x_proposed = compute_cost_fcn(x_proposed, weights_proposed, residuals, args...);

            if (params.GNC_verbose)
            {        std::cout << "{Inner loop} Previous cost: " << f_x << std::endl;
                     std::cout << "{Inner loop} Current cost: " << f_x_proposed << std::endl;
            }
            diff_costs = fabs(f_x - f_x_proposed);
            if (params.GNC_verbose)           std::cout << "{Inner loop} Diff between previous cost and current one: " << diff_costs << std::endl;


            // Update variables
            x = std::move(x_proposed);
            f_x = f_x_proposed;
            weights = std::move(weights_proposed);

            // Record values if required
            if (params.log_outer_iters)
            {
              outer_results.objective_values.push_back(f_x);
              outer_results.time_inner_iter.push_back(10);  // TODO
              if (params.log_outer_iters)         outer_results.iterates_x.push_back(x);
              outer_results.diff_costs_updates.push_back(diff_costs);
            }

            if (fabs(diff_costs) < params.cost_diff_threshold)
            {
              if (params.GNC_verbose)           std::cout << "{Inner loop} Cost convergence. Breaking inner loop\n";
              if (params.log_outer_iters)       outer_results.GNCouterstatus = GNCOuterStatus::CONVERGENCE_COST;

              break;  // end inner loop
            }
        }  // end each inner iteration

        // 5. Update mu
        if (params.GNC_verbose)           std::cout << "{Outer loop} Updating mu\n";
        mu_proposed = update_mu_fcn(mu, params, args...);

        // end of outer loop
        // std::cout << weights << std::endl;
        // Record output
        outer_results.x = x;
        outer_results.f = f_x;
        outer_results.weights = weights;
        outer_results.mu = mu;
        outer_results.diff_costs = diff_costs;
        outer_results.elapsed_time = 10;  // TOD
        outer_results.nr_inner_iterations = i_inner_iter;

        // record outer iteration in final struct
        if (params.log_outer_iters)
        {
          result_gnc.intermediate_outer_results.push_back(outer_results);
        }
        mu = mu_proposed;

        // 6. Check conditions
        if (params.GNC_verbose)
        {
          std::cout << "{Outer loop} Previous cost: " << f_outer_prev << std::endl;
          std::cout << "{Outer loop} Current cost: " << f_x << std::endl;
        }
        if (fabs(f_outer_prev - f_x) < params.cost_diff_threshold)
          {
            if (params.GNC_verbose)           std::cout << "{Outer loop} Differencen between previous outer cost and current = " 
                                                << fabs(f_outer_prev - f_x) << ".\nBreaking outer loop\n";
            result_gnc.GNCstatus = GNCStatus::CONVERGENCE_COST;
            break;
          }
        // Update cost for the last outer loop
        f_outer_prev = f_x;

      } // end outer iterations
    }  // end else (abs(mu) == 0)


if (params.GNC_verbose)           std::cout << "Finding inliers\n";
// Find inliers
set_inliers.setZero();  // clear
for (size_t i = 0; i < n; i++)
  set_inliers(i) = (weights(i) > params.inliers_threshold);

if (set_inliers.sum() < params.nr_min_points)     std::cout << "We obtained less points than you need for the estimation!!\n";

// Estimate variable one last time
// 1.0 Update params
if (update_params_in_fcn)
{
  (*update_params_in_fcn)(x, set_inliers, args...);
}
// 1.1 Initialize x if required
  if (initialize_variable_fcn)
  {
    x = (*initialize_variable_fcn)(x, set_inliers, args...);
  }

// 1.3 Run actual solver!
if (params.GNC_verbose)           std::cout << "Running one last time the variable estimator...\n";
x_proposed = estimate_model_fcn(x, set_inliers, args...);

// compute the associated cost
// 1. residuals
residuals = compute_residuals_fcn(x_proposed, args...);
// 2. cost
f_x_proposed = compute_cost_fcn(x_proposed, set_inliers, residuals, args...);

if ((result_gnc.GNCstatus == GNCStatus::MU_ZERO) || (result_gnc.GNCstatus == GNCStatus::CONVERGENCE_COST) || (result_gnc.GNCstatus == GNCStatus::USER_FUNCTION))
{
  // Save last results
  result_gnc.x = std::move(x_proposed);
  result_gnc.f = f_x_proposed;
  result_gnc.weights = std::move(weights);
  result_gnc.set_inliers = std::move(set_inliers);
  result_gnc.mu = mu;
  result_gnc.elapsed_time = 100;  // TODO: DO this
  result_gnc.nr_outer_iterations = i_outer_iter;
}
  return result_gnc;
} // end function


/*   GM    */

template <typename Variable, typename Weights, typename Scalar=double, typename... Args>
GNCResult<Variable, Weights, Scalar>
GMGNC( const VariableEstimation<Variable, Weights, Args...>& estimate_model_fcn,
     const ComputeCost<Variable, Weights, Scalar, Args...>& compute_cost_fcn,
     const ComputeResiduals<Variable, Weights, Args...>& compute_residuals_fcn,
     const Variable& initial_estimation_x,
     const Weights& initial_weights,
     Args &... args,
     const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
     const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
     const GNCParams<Scalar> & params = GNCParams<Scalar>())
     {
       // Run GNC using GM
       return GNC<Variable, Weights, Scalar, Args...>
       (estimate_model_fcn, compute_cost_fcn, compute_residuals_fcn,
       GM::GMInitializationMu<Weights, Scalar, Args...>, GM::GMUpdateWeights<Variable, Scalar, Weights, Args...>,
       GM::GMUpdateMu<Scalar, Args...>, initial_estimation_x, initial_weights, args..., update_params_in_fcn, initialize_variable_fcn, params);
     }

/*   TLS    */

     template <typename Variable, typename Weights, typename Scalar=double, typename... Args>
     GNCResult<Variable, Weights, Scalar>
     TLSGNC( const VariableEstimation<Variable, Weights, Args...>& estimate_model_fcn,
          const ComputeCost<Variable, Weights, Scalar, Args...>& compute_cost_fcn,
          const ComputeResiduals<Variable, Weights, Args...>& compute_residuals_fcn,
          const Variable& initial_estimation_x,
          const Weights& initial_weights,
          Args &... args,
          const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
          const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
          const GNCParams<Scalar> & params = GNCParams<Scalar>())
          {
            // Run GNC using GM
            return GNC<Variable, Weights, Scalar, Args...>
            (estimate_model_fcn, compute_cost_fcn, compute_residuals_fcn,
            TLS::TLSInitializationMu<Weights, Scalar, Args...>, TLS::TLSUpdateWeights<Variable, Scalar, Weights, Args...>,
            TLS::TLSUpdateMu<Scalar, Args...>, initial_estimation_x, initial_weights, args..., update_params_in_fcn, initialize_variable_fcn, params);
          }
};  // end of gncso namespace
