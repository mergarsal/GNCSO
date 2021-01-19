
#pragma once


#include "gncso/Base/Concepts.h"
#include <iostream>   // std::cout, std::fixed, std::scientific

#define LN10 2.30258509299

namespace gncso {

  namespace Tukey{


template <typename Weights, typename Scalar = double, typename ...Args>
Scalar TukeyInitializationMu(const Weights & residuals, const GNCParams<Scalar> & params, Args &... args)
{

  return (100000);
}


template <typename Variable, typename Scalar = double, typename Weights, typename... Args>
Weights TukeyUpdateWeights(const Variable &X, const Weights& prev_weights,
                              const Weights& residuals, const Scalar& mu, const GNCParams<Scalar> & params, Args &... args)
{
  
  Weights weights;
  size_t N = residuals.size();
  weights.resize(N);
  
  
  for (size_t i = 0; i < N; i++)
  {
    
    Scalar r_sq_mu_c2 = residuals(i) / (mu * params.max_res_tol_sq);
    
    if (r_sq_mu_c2 > 1) weights(i) = 0.0;
    else weights(i) = pow((1 - r_sq_mu_c2), 2);
     
    // fix the limits to the weights
    if (weights(i) < 0.0)   weights(i) = 0.0;
    if (weights(i) > 1.0)   weights(i) = 1.0;
  }


  return weights;
}


 template <typename Scalar = double, typename... Args>
Scalar TukeyUpdateMu(const Scalar& prev_mu, const GNCParams<Scalar> & params, Args &... args)
{
  Scalar mu = prev_mu / params.gnc_factor;
  return (mu < 1.0 ? 1.0 : mu);
}

} // end namespace of Tukey

}  // end namespace of gnso
