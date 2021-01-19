#pragma once


#include "gncso/Base/Concepts.h"
#include <iostream>   // std::cout, std::fixed, std::scientific

#define LN10 2.30258509299

namespace gncso {

  namespace Welsch{


template <typename Weights, typename Scalar = double, typename ...Args>
Scalar WelschInitializationMu(const Weights & residuals, const GNCParams<Scalar> & params, Args &... args)
{

  return (1000000);
}


template <typename Variable, typename Scalar = double, typename Weights, typename... Args>
Weights WelschUpdateWeights(const Variable &X, const Weights& prev_weights,
                              const Weights& residuals_sq, const Scalar& mu, const GNCParams<Scalar> & params, Args &... args)
{
  
  Weights weights;
  size_t N = residuals_sq.size();
  weights.resize(N);
   
  
  for (size_t i = 0; i < N; i++)
  {
    weights(i) = std::exp(-residuals_sq(i) / (mu * params.max_res_tol_sq));
    // fix the limits to the weights
    if (weights(i) < 0.0)   weights(i) = 0.0;
    if (weights(i) > 1.0)   weights(i) = 1.0;
  }


  return weights;
}


 template <typename Scalar = double, typename... Args>
Scalar WelschUpdateMu(const Scalar& prev_mu, const GNCParams<Scalar> & params, Args &... args)
{
  Scalar mu = prev_mu / params.gnc_factor;
  return (mu < 1.0 ? 1.0 : mu);
}

} // end namespace of Welsch

}  // end namespace of gnso
