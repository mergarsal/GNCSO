
#pragma once


#include "gncso/Base/Concepts.h"
#include <iostream>   // std::cout, std::fixed, std::scientific

namespace gncso {

  namespace GM{


template <typename Weights, typename Scalar = double, typename ...Args>
Scalar GMInitializationMu(const Weights & residuals, const GNCParams<Scalar> & params, Args &... args)
{
  std::cout << "[GM] Initializing mu...\n";

  Scalar mu = residuals.maxCoeff() / (params.max_res_tol_sq);

  // cases
  if (mu < 1.0)           mu = 1.0;
  if (mu > 2000)         mu = 7000;  
  return (mu);
}


template <typename Variable, typename Scalar = double, typename Weights, typename... Args>
Weights GMUpdateWeights(const Variable &X, const Weights& prev_weights,
                              const Weights& residuals, const Scalar& mu, const GNCParams<Scalar> & params, Args &... args)
{

  Weights weights;
  size_t N = residuals.size();
  weights.resize(N);
  
  for (size_t i = 0; i < N; i++)
  {
    Scalar den_frac = residuals(i) + mu * params.max_res_tol_sq;
    if (fabs(den_frac) < 1e-09)
        weights(i) = 1.0;
    else
      weights(i) = pow( (mu * params.max_res_tol_sq) / (den_frac), 2);
    if (weights(i) < 0.0)   weights(i) = 0.0;
    if (weights(i) > 1.0)   weights(i) = 1.0;
  }

  return weights;
}


 template <typename Scalar = double, typename... Args>
Scalar GMUpdateMu(const Scalar& prev_mu, const GNCParams<Scalar> & params, Args &... args)
{
  Scalar mu = prev_mu / params.gnc_factor;
  return (mu < 1.0 ? 1.0 : mu);
}

} // end namespace of GM

}  // end namespace of gnso
