#pragma once

// #include <algorithm>
// #include <cmath>
#include <experimental/optional>
#include <functional>
#include <iostream>
// #include <limits>


#include "gncso/Refinement/Base/Concepts.h"

// for the general fcns
using namespace baseOpt;

namespace refinements
{
  // TODO: include help

  template <typename Variable, typename Weights, typename Tangent, typename Scalar=double, typename... Args>
  RefinedResult<Variable, Weights, Scalar> 
  expandSetInliers (  
       const Optimization::Objective<Variable, Scalar, Args...> &f,
       const Optimization::Riemannian::QuadraticModel<Variable, Tangent, Args...> &QM,
       const Optimization::Riemannian::RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
       const Optimization::Riemannian::Retraction<Variable, Tangent, Args...> &retract,
       const baseOpt::ComputeResiduals<Variable, Weights, Args...>& compute_residuals_inliers_fcn,
       const ComputeSetInliers<Variable, Weights, Scalar, Args...>& compute_set_inliers_fcn,
       const ComputeMaxCoefficient<Variable, Weights, Scalar, Args...>& compute_max_coeff_fcn,
       const Variable& initial_estimation_x,
       const Weights& initial_set_inliers,
       Args &... args,
       const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
       const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
       const std::experimental::optional<Optimization::Riemannian::LinearOperator<Variable, Tangent, Args...>> &precon = std::experimental::nullopt,
       const RefinParams<Scalar> & params = RefinParams<Scalar>(),
       const std::experimental::optional<Optimization::Riemannian::TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
          std::experimental::nullopt)
{
  std::cout << "--------------- Inside Expand set of inliers function ---------------\n";


  // output struct
  RefinedResult<Variable, Weights, Scalar> result_refinement;
  result_refinement.refin_status = RefinStatus::ELAPSED_TIME;  // default

  // Current iterate and proposed next iterates;
  Variable x, x_proposed, x_former;

  // Function value at the current iterate and proposed iterate
  Scalar f_x, f_x_proposed, f_outer_prev;
  f_x = 10000000;  // Initial value
  f_outer_prev = f_x;

  // Set of inliers at the current iterate
  Weights set_inliers, set_inliers_proposed;

  // Residuals at the current iterate
  Weights residuals;

  // Initialize Variable
  x = initial_estimation_x;
  // Initialize set of inliers
  set_inliers = initial_set_inliers;

  //  Update parameters if needed
  if (update_params_in_fcn)
  {
        (*update_params_in_fcn)(initial_estimation_x, initial_set_inliers, args...);
  }



  size_t i_iter = 0;  // to make it visible outside the loop
  Scalar diff_costs;
  size_t n_inliers_old = 0, n_inliers = 0;  // number of inliers in the former iteration
  Scalar max_res = 10000000;
  Scalar upper_bound_res = 1000000;

  // star expansion
  for (i_iter = 0; i_iter < params.max_iters; i_iter++)
      {
        std::cout << "Iteration : " << i_iter << std::endl;

        // 1. Compute residuals for set of inliers
        residuals = compute_residuals_inliers_fcn(x, args...);

        // std::cout << "Residuals:\n" << residuals << std::endl;
        // 2. Get max residuals
        max_res = compute_max_coeff_fcn(x, set_inliers, residuals, args...);
        std::cout << "Max residual = " << max_res << std::endl;

        // 3. save number of inliers
        n_inliers_old = set_inliers.sum();
        
        std::cout << "Number of inliers: " << n_inliers_old << std::endl;

        // 0. Initialize variable if required
        // INFO: no se si poner esto aqui o antes del R-TNT
        // if (initialize_variable_fcn)
        //{
        //   x = (*initialize_variable_fcn)(x, set_inliers, args...);
        //}

        // 3. Compute new upper bound for residuals
        upper_bound_res = max_res * params.factor_ref_residual;
        std::cout << "Upper bound for the residuals: " << upper_bound_res << std::endl;

        // 2. Compute new set of inliers
        // b. Compute new set
        set_inliers_proposed = compute_set_inliers_fcn(x, set_inliers, residuals,
                    upper_bound_res, args...);
        n_inliers = set_inliers_proposed.sum();
        
        std::cout << "Number of inliers after expanding: " << n_inliers << std::endl;

        // 3. Check cardinality of the new set of inliers
        if (n_inliers <= n_inliers_old)
        {
          // here we keep:
          // set_inliers
          // and the x from the previos iteration
          std::cout << "We have now fewer inliers than before.\nBreaking the loop...\n";
          result_refinement.refin_status = RefinStatus::CONVERGENCE_SET;
          break;
        }

        std::cout << "Updating the params\n";
        //  Update parameters if needed
        if (update_params_in_fcn)
        {
              (*update_params_in_fcn)(x, set_inliers_proposed, args...);
        }

        // else
        std::cout << "Estimating the new variable\n";
        // 3. Estimate new variable as a function of the new set
        Optimization::Riemannian::TNTResult<Variable, Scalar> TNTResults = Optimization::Riemannian::TNT<Variable, Tangent, Scalar, Args...>(f, QM, metric, retract, x, args..., precon, params, user_function);

        x_proposed = TNTResults.x;

        // Update variables
        x = std::move(x_proposed);
        set_inliers = std::move(set_inliers_proposed);

      }  // end for i = 1, ..., max_iters

      // save results
      if (i_iter == params.max_iters)
        result_refinement.refin_status = RefinStatus::MAX_NR_ITERS;

      result_refinement.x = std::move(x);
      result_refinement.set_inliers = std::move(set_inliers);
      result_refinement.elapsed_time = 100000;
      result_refinement.nr_iterations = i_iter;
      result_refinement.upper_bound_res = upper_bound_res;
      
      // std::cout << "Final set of inliers:\n" << result_refinement.set_inliers << std::endl; 

      // exit fcn
      return result_refinement;

}

/* Basic inlier selection */
/* Variable estimation */
template <typename Variable, typename Weights, typename Scalar, typename... Args>
Weights BasicComputeSetInliers(const Variable &X, const Weights& inliers,
         const Weights& residuals, const Scalar upper_bound_residuals, Args &... args)
{
   assert(inliers.rows() == residuals.rows());
   
   Weights new_inliers(inliers.size()); 
   new_inliers.setZero();
   
   for(size_t i=0; i < inliers.rows(); i++)
   {
        new_inliers(i) = (residuals(i) <= upper_bound_residuals) ? 1 : 0;
        // std::cout << "Residual: " << residuals(i) << std::endl;
   }
   
   return new_inliers;
                  
}

/* Basic max coeff. computation */
/* Variable estimation */
template <typename Variable, typename Weights, typename Scalar, typename... Args>
Scalar BasicComputeMaxCoefficient(const Variable &X, const Weights& inliers,
         const Weights& residuals, Args &... args)
{
   assert(inliers.cols() == residuals.cols());
   
   // std::cout << "cWiseProduct:\n" << residuals.cwiseProduct(inliers) << std::endl;
   return (residuals.cwiseProduct(inliers)).maxCoeff();
                  
}





  /* Refinement with basic inlier selection */ 
  
  template <typename Variable, typename Weights, typename Tangent, typename Scalar=double, typename... Args>
  RefinedResult<Variable, Weights, Scalar> 
  basicExpandSetInliers(  
       const Optimization::Objective<Variable, Scalar, Args...> &f,
       const Optimization::Riemannian::QuadraticModel<Variable, Tangent, Args...> &QM,
       const Optimization::Riemannian::RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
       const Optimization::Riemannian::Retraction<Variable, Tangent, Args...> &retract,
       const baseOpt::ComputeResiduals<Variable, Weights, Args...>& compute_residuals_inliers_fcn,
       const Variable& initial_estimation_x,
       const Weights& initial_set_inliers,
       Args &... args,
       const std::experimental::optional<UpdateParamsInProblem<Variable, Scalar, Weights, Args...>>& update_params_in_fcn = std::experimental::nullopt,
       const std::experimental::optional<InitializeVariableX<Variable, Scalar, Weights, Args...>> &initialize_variable_fcn = std::experimental::nullopt,
       const std::experimental::optional<Optimization::Riemannian::LinearOperator<Variable, Tangent, Args...>> &precon = std::experimental::nullopt,
       const RefinParams<Scalar> & params = RefinParams<Scalar>(),
       const std::experimental::optional<Optimization::Riemannian::TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
          std::experimental::nullopt)

{
        return expandSetInliers<Variable, Weights, Tangent, Scalar, Args...> (f, QM, metric, retract, compute_residuals_inliers_fcn, BasicComputeSetInliers<Variable, Weights, Scalar, Args...>, BasicComputeMaxCoefficient<Variable, Weights, Scalar, Args ...>, initial_estimation_x, initial_set_inliers, args..., update_params_in_fcn, initialize_variable_fcn, precon, params, user_function);
}

};  // end of namespace refinements
