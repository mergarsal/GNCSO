#pragma once


#include <algorithm>
#include <cmath>
#include <experimental/optional>
#include <functional>
#include <iostream>
#include <limits>

// GNC
#include "gncso/Base/Concepts.h"
#include "gncso/GM/GM_fcn.h"
#include "gncso/TLS/TLS_fcn.h"
#include "gncso/Temperature/temperature.h"
#include "gncso/Welsch/Welsch_fcn.h"
#include "gncso/Tukey/Tukey_fcn.h"

// Smooth
// #include "Optimization/Convex/Concepts.h"
#include "Optimization/Riemannian/TNT.h"
#include "Optimization/Riemannian/Concepts.h"

// Smooth
using namespace Optimization;
using namespace Optimization::Riemannian;
using namespace Optimization::Convex;

// for the general fcns
using namespace baseOpt;




// GNC
namespace gncso
{

template <typename Scalar = double>
struct GNCSmoothParams : public TNTParams<Scalar> , public GNCParams<Scalar> {
GNCSmoothParams() : TNTParams<Scalar>(), GNCParams<Scalar>() {};
};

/* Basic for update inner mu */
template <typename Scalar = double, typename... Args>
Scalar baseUpdateMuInner(const Scalar& mu, const Scalar& id_inner, const GNCParams<Scalar> & params, Args &... args)
{
        return (mu / (id_inner+ 1));
}


template <typename Variable, typename Weights, typename Tangent, typename Scalar=double, typename... Args>
GNCResult<Variable, Weights, Scalar>
 GNCSmooth(const Optimization::Objective<Variable, Scalar, Args...> &f,
     const Optimization::Riemannian::QuadraticModel<Variable, Tangent, Args...> &QM,
     const Optimization::Riemannian::RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
     const Optimization::Riemannian::Retraction<Variable, Tangent, Args...> &retract,
       const ComputeResiduals<Variable, Weights, Args...>& compute_residuals_fcn_sq,
       const InitializationMu<Weights, Scalar, Args...>& initialize_mu_fcn,
       const UpdateWeights<Variable, Scalar, Weights, Args...>& update_weights_fcn,
       const UpdateMu<Scalar, Args...>& update_mu_fcn,
       const  UpdateMuInner<Scalar, Args...>& update_mu_inner_fcn,
       const Variable& initial_estimation_x,
       const Weights& initial_weights,
       Args &... args,
       const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
       const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
       const std::experimental::optional<Optimization::Riemannian::LinearOperator<Variable, Tangent, Args...>> &precon = std::experimental::nullopt,
       const GNCSmoothParams<Scalar> & params = GNCSmoothParams<Scalar>(),
       const std::experimental::optional<Optimization::Riemannian::TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
          std::experimental::nullopt)
         {
           std::cout << "---------------Inside GNC-RTNT function---------------\n";
           // Output struct

           GNCResult<Variable, Weights, Scalar> result_gnc;
           result_gnc.GNCstatus = GNCStatus::ITERATION_LIMIT;

           // Current iterate and proposed next iterates;
           Variable x, x_proposed;

           // Function value at the current iterate and proposed iterate
           Scalar f_x, f_x_proposed, f_outer_prev;
           f_x = 10000000;  // Initial value
           f_outer_prev = f_x;

           Scalar mu, mu_proposed, mu_weight;

           // Weights at the current iterate
           Weights weights, weights_proposed;

           // residuals_sq at the current iterate
           Weights residuals_sq, prev_residuals_sq;

           size_t n = initial_weights.rows();
           Weights set_inliers;
           set_inliers.resize(n);


           // Initialize Variable
           x = initial_estimation_x;
           // Initialize weights
           weights = initial_weights;
           weights_proposed = initial_weights;
           size_t i_outer_iter = 0;  // to make it visible outside the loop
           size_t i_inner_iter = 0;  // to make it visible outside the loop
           Scalar diff_costs;

           // Setting up initial weights
           if (update_params_in_fcn)
           {
             (*update_params_in_fcn)(initial_estimation_x, initial_weights, args...);
           }
           
           // 1.2 Run actual optimization
          Optimization::Riemannian::TNTResult<Variable, Scalar> TNTResults_pre = Optimization::Riemannian::TNT<Variable, Tangent, Scalar, Args...>(f, QM, metric, retract, initial_estimation_x, args..., precon, params, user_function);

          // 1.3 Extract solution
          x_proposed = TNTResults_pre.x;


           // compute residuals
           residuals_sq = compute_residuals_fcn_sq(x_proposed, args...);

           // initialize mu
           mu = initialize_mu_fcn(residuals_sq, params, args...);
           if (params.GNC_verbose)    std::cout << "Mu initialized to " << mu << std::endl;


         
              x = std::move(x_proposed);
              weights = std::move(weights_proposed);
              f_x = f_x_proposed;

        
           {
             

             for (i_outer_iter = 0; i_outer_iter < params.max_outer_iterations; i_outer_iter++)
             {

               GNCOuterResult<Variable, Weights, Scalar> outer_results;

               // Start of outer loop
               if (params.log_outer_iters)       outer_results.GNCouterstatus = GNCOuterStatus::MAX_NR_ITERS;
    

               // Run inner loop
               for (i_inner_iter = 0; i_inner_iter < params.max_inner_iterations; i_inner_iter++)
               {
                 if (params.GNC_verbose)
                 {
                    std::cout << "Outer iter: " << i_outer_iter << "               mu: " << mu;
                    std::cout << "                    Inner iter: " << i_inner_iter << std::endl;
                 }

                   // 1. Fix weights and estimate E
                   // 1.2 Run actual optimization
                   Optimization::Riemannian::TNTResult<Variable, Scalar> TNTResults = Optimization::Riemannian::TNT<Variable, Tangent, Scalar, Args...>(f, QM, metric, retract, x, args..., precon, params, user_function);


                   // 1.3 Extract solution
                   x_proposed = TNTResults.x;
                   // 2. Compute residuals_sq

                   // save previous residuals
                   prev_residuals_sq = residuals_sq;

                   residuals_sq = compute_residuals_fcn_sq(x_proposed, args...);


                   // 3. Fix X and compute weights

                   // update mu for inner for

                   mu_weight = update_mu_inner_fcn(mu, i_inner_iter, params, args...);
                   // standard update
 

                   weights_proposed = update_weights_fcn(x_proposed, weights, residuals_sq, mu_weight, params, args...);

                   // 3.2. Update weights
                   // this function gives you the opportunity
                   // to update any variable inside the problem!!
                   if (update_params_in_fcn)
                   { (*update_params_in_fcn)(x_proposed, weights_proposed, args...);  }

                   // 4. Compute cost
                   f_x_proposed = f(x_proposed, args...);
                   diff_costs = fabs(f_x - f_x_proposed);

                   double sum_abs = 0;
                   for (size_t j = 0; j < n; j++)
                   {
                        sum_abs += fabs(prev_residuals_sq(j) - residuals_sq(j));
                   }
                   


                   Scalar n_inliers_before = set_inliers.sum();

                   for (size_t i = 0; i < n; i++)
                     set_inliers(i) = (weights_proposed(i) > params.inliers_threshold);
                   
             
                   
                   if (set_inliers.sum() < params.nr_min_points )
                   {
                       if (params.GNC_verbose)  std::cout << "[WARNING] Run out of data!!\n" << "We have " << set_inliers.sum() << " inliers!";
                       // we have less points than required!!
                       break;                       

                   }


                   if (params.GNC_verbose)
                   {
                           std::cout << "{Inner loop} Previous cost: " << f_x << std::endl;
                           std::cout << "{Inner loop} Current cost: " << f_x_proposed << std::endl;
                           std::cout << "{Inner loop} Diff between previous cost and current one: " << diff_costs << std::endl;
                   }

                   // Update variables
                   x = std::move(x_proposed);
                   f_x = f_x_proposed;
                   weights = std::move(weights_proposed);

                   // Record values if required
                   if (params.log_outer_iters)
                   {
                     outer_results.objective_values.push_back(f_x);
                     outer_results.time_inner_iter.push_back(10);  // TODO
                     if (params.GNClog_iterates)
                       outer_results.iterates_x.push_back(x);
                     outer_results.diff_costs_updates.push_back(diff_costs);
                   }

                   if (fabs(diff_costs) < params.cost_diff_threshold)
                   {
                     if (params.GNC_verbose)   std::cout << "{Inner loop} Cost convergence. Breaking inner loop\n";
                     if (params.log_outer_iters)     outer_results.GNCouterstatus = GNCOuterStatus::CONVERGENCE_COST;

                     break;  // end inner loop
                   }


                 }  // end each inner iteration


               // 5. Update mu
               if (params.GNC_verbose) std::cout << "{Outer loop} Updating mu\n";
               mu_proposed = update_mu_fcn(mu, params, args...);

               // end of outer loop

               // Record output
               outer_results.x = x;
               outer_results.f = f_x;
               outer_results.weights = weights;
               outer_results.mu = mu;
               outer_results.diff_costs = diff_costs;
               outer_results.elapsed_time = 10;  // TODO
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
                   if (params.GNC_verbose) std::cout << "{Outer loop} Differencen between previos outer cost and current = " << fabs(f_outer_prev - f_x) << ".\nBreaking outer loop\n";
                   result_gnc.GNCstatus = GNCStatus::CONVERGENCE_COST;
                   break;
                 }
               // Update cost for the last outer loop
               f_outer_prev = f_x;
             } // end outer iterations

         }  // end of if: mu != 0


          std::cout << "Finding inliers last round\n";
         if (params.GNC_verbose) std::cout << "Finding inliers\n";
         // Find inliers
         set_inliers.setZero();
         for (size_t i = 0; i < n; i++)
         set_inliers(i) = (weights(i) > params.inliers_threshold);
         std::cout << "Number of inliers: " << set_inliers.sum() << std::endl;

         if (set_inliers.sum() < params.nr_min_points)
         {
                std::cout << "We obtained less points than you need for the estimation!!\n";
                result_gnc.valid_estimation = false;
         }
         else   result_gnc.valid_estimation = true;

         if (params.GNC_verbose) std::cout << "Running one last time the R-TNT\n";
         // Estimate variable one last time
         // 1.1 Update problem with set_inliers
         if (update_params_in_fcn)
         {    (*update_params_in_fcn)(x, set_inliers, args...);}
         // 1.2. Re-initialize if needed
         // before calling the solver, you can compute your own Initialization
         if (initialize_variable_fcn)
         {
           x = (*initialize_variable_fcn)(x, weights, args...);
         }
         // 1.3 Run solver!
         Optimization::Riemannian::TNTResult<Variable, Scalar> TNTResults = Optimization::Riemannian::TNT<Variable, Tangent, Scalar, Args...>(
                      f, QM, metric, retract, x, args..., precon, params, user_function);

         // 1.3 Extract solution
         x_proposed = TNTResults.x;

         // Compute cost value
         f_x_proposed = f(x_proposed, args...);

         if ((result_gnc.GNCstatus == GNCStatus::MU_ZERO) || (result_gnc.GNCstatus == GNCStatus::CONVERGENCE_COST) || (result_gnc.GNCstatus == GNCStatus::USER_FUNCTION))
         {
           // Save last results
           result_gnc.x = std::move(x_proposed);
           result_gnc.f = f_x_proposed;
           result_gnc.weights = std::move(weights);
           result_gnc.set_inliers = std::move(set_inliers);
           result_gnc.mu = mu;
           result_gnc.elapsed_time = 100;  // TODO:
           result_gnc.nr_outer_iterations = i_outer_iter;
         }

         return result_gnc;
}


/*   GM    */

        template <typename Variable, typename Weights, typename Tangent, typename Scalar=double, typename... Args>
        GNCResult<Variable, Weights, Scalar>
         GMGNCSmooth(    const Optimization::Objective<Variable, Scalar, Args...> &f,
                         const Optimization::Riemannian::QuadraticModel<Variable, Tangent, Args...> &QM,
                         const Optimization::Riemannian::RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
                         const Optimization::Riemannian::Retraction<Variable, Tangent, Args...> &retract,
                         const ComputeResiduals<Variable, Weights, Args...>& compute_residuals_fcn_sq,
                         const Variable& initial_estimation_x,
                         const Weights& initial_weights,
                         Args &... args,
                         const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
                         const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
                         const std::experimental::optional<Optimization::Riemannian::LinearOperator<Variable, Tangent, Args...>> &precon = std::experimental::nullopt,
                         const GNCSmoothParams<Scalar> & params = GNCSmoothParams<Scalar>(),
                         const std::experimental::optional<Optimization::Riemannian::TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
                            std::experimental::nullopt)

            {
              std::cout << "\n---------------Inside GNC-GM   function---------------\n";
              // Run GNC using GM
              return GNCSmooth<Variable, Weights, Tangent, Scalar, Args...>  (f, QM, metric, retract, compute_residuals_fcn_sq,
              GM::GMInitializationMu<Weights, Scalar, Args...>, GM::GMUpdateWeights<Variable, Scalar, Weights, Args...>,
              GM::GMUpdateMu<Scalar, Args...>, baseUpdateMuInner<Scalar, Args...>, initial_estimation_x, initial_weights, args..., update_params_in_fcn,
              initialize_variable_fcn, precon, params, user_function);
            }



            /*   TLS    */

            template <typename Variable, typename Weights, typename Tangent, typename Scalar=double, typename... Args>
            GNCResult<Variable, Weights, Scalar>
             TLSGNCSmooth(    const Optimization::Objective<Variable, Scalar, Args...> &f,
                             const Optimization::Riemannian::QuadraticModel<Variable, Tangent, Args...> &QM,
                             const Optimization::Riemannian::RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
                             const Optimization::Riemannian::Retraction<Variable, Tangent, Args...> &retract,
                             const ComputeResiduals<Variable, Weights, Args...>& compute_residuals_fcn_sq,
                             const Variable& initial_estimation_x,
                             const Weights& initial_weights,
                             Args &... args,
                             const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
                             const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
                             const std::experimental::optional<Optimization::Riemannian::LinearOperator<Variable, Tangent, Args...>> &precon = std::experimental::nullopt,
                             const GNCSmoothParams<Scalar> & params = GNCSmoothParams<Scalar>(),
                             const std::experimental::optional<Optimization::Riemannian::TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
                                std::experimental::nullopt)

                {
                  std::cout << "\n---------------Inside GNC-TLS  function---------------\n";
                  // Run GNC using GM
                  return GNCSmooth<Variable, Weights, Tangent, Scalar, Args...>  (f, QM, metric, retract, compute_residuals_fcn_sq,
                  TLS::TLSInitializationMu<Weights, Scalar, Args...>, TLS::TLSUpdateWeights<Variable, Scalar, Weights, Args...>,
                  TLS::TLSUpdateMu<Scalar, Args...>, baseUpdateMuInner<Scalar, Args...>, initial_estimation_x, initial_weights, args..., update_params_in_fcn,
                  initialize_variable_fcn, precon, params, user_function);
                }




 /*   Welsch    */

            template <typename Variable, typename Weights, typename Tangent, typename Scalar=double, typename... Args>
            GNCResult<Variable, Weights, Scalar>
            WelschGNCSmooth(    const Optimization::Objective<Variable, Scalar, Args...> &f,
                             const Optimization::Riemannian::QuadraticModel<Variable, Tangent, Args...> &QM,
                             const Optimization::Riemannian::RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
                             const Optimization::Riemannian::Retraction<Variable, Tangent, Args...> &retract,
                             const ComputeResiduals<Variable, Weights, Args...>& compute_residuals_fcn_sq,
                             const Variable& initial_estimation_x,
                             const Weights& initial_weights,
                             Args &... args,
                             const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
                             const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
                             const std::experimental::optional<Optimization::Riemannian::LinearOperator<Variable, Tangent, Args...>> &precon = std::experimental::nullopt,
                             const GNCSmoothParams<Scalar> & params = GNCSmoothParams<Scalar>(),
                             const std::experimental::optional<Optimization::Riemannian::TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
                                std::experimental::nullopt)

                {
                  std::cout << "\n---------------Inside GNC-Welsch  function---------------\n";
                  // Run GNC using GM
                  return GNCSmooth<Variable, Weights, Tangent, Scalar, Args...>  (f, QM, metric, retract, compute_residuals_fcn_sq,
                  Welsch::WelschInitializationMu<Weights, Scalar, Args...>, Welsch::WelschUpdateWeights<Variable, Scalar, Weights, Args...>,
                  Welsch::WelschUpdateMu<Scalar, Args...>, baseUpdateMuInner<Scalar, Args...>, initial_estimation_x, initial_weights, args..., update_params_in_fcn,
                  initialize_variable_fcn, precon, params, user_function);
                }


/*   Tukey   */

            template <typename Variable, typename Weights, typename Tangent, typename Scalar=double, typename... Args>
            GNCResult<Variable, Weights, Scalar>
            TukeyGNCSmooth(    const Optimization::Objective<Variable, Scalar, Args...> &f,
                             const Optimization::Riemannian::QuadraticModel<Variable, Tangent, Args...> &QM,
                             const Optimization::Riemannian::RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
                             const Optimization::Riemannian::Retraction<Variable, Tangent, Args...> &retract,
                             const ComputeResiduals<Variable, Weights, Args...>& compute_residuals_fcn_sq,
                             const Variable& initial_estimation_x,
                             const Weights& initial_weights,
                             Args &... args,
                             const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
                             const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
                             const std::experimental::optional<Optimization::Riemannian::LinearOperator<Variable, Tangent, Args...>> &precon = std::experimental::nullopt,
                             const GNCSmoothParams<Scalar> & params = GNCSmoothParams<Scalar>(),
                             const std::experimental::optional<Optimization::Riemannian::TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
                                std::experimental::nullopt)

                {
                  std::cout << "\n---------------Inside GNC-Tukey  function---------------\n";
                  // Run GNC using GM
                  return GNCSmooth<Variable, Weights, Tangent, Scalar, Args...>  (f, QM, metric, retract, compute_residuals_fcn_sq,
                  Tukey::TukeyInitializationMu<Weights, Scalar, Args...>, Tukey::TukeyUpdateWeights<Variable, Scalar, Weights, Args...>,
                  Tukey::TukeyUpdateMu<Scalar, Args...>, baseUpdateMuInner<Scalar, Args...>, initial_estimation_x, initial_weights, args..., update_params_in_fcn,
                  initialize_variable_fcn, precon, params, user_function);
                }

};  // end of gncso namespace
