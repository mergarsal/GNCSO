#pragma once


#include "gncso/Base/Concepts.h"
#include <iostream>  // std::cout, std::fixed, std::scientific

namespace gncso {

namespace TLS{


template <typename Weights, typename Scalar = double, typename ...Args>
Scalar TLSInitializationMu(const Weights & residuals, const GNCParams<Scalar> & params, Args &... args)
{
  return (1 / (2 * residuals.maxCoeff() / params.max_res_tol_sq - 1));
}


template <typename Variable, typename Scalar = double, typename Weights, typename... Args>
Weights TLSUpdateWeights(const Variable &X, const Weights& prev_weights,
                              const Weights& residuals, const Scalar& mu, const GNCParams<Scalar> & params, Args &... args)
{

  size_t N = residuals.size();

  // Output variable
  Weights weights;
  weights.resize(N);

  // limits
  Scalar th1, th2;
  th1 = (mu + 1) / mu * params.max_res_tol_sq;
  th2 = mu / (mu + 1) *  params.max_res_tol_sq;

  // for each observation
  for (size_t i = 0; i < N; i++)
  {
    if (residuals(i) >= th1)
            weights(i) = 0.0;
    else if (residuals(i) <= th2)
            weights(i) = 1.0;
    else
    {
            if (fabs(residuals(i)) < 1e-08)
              weights(i) = 1.0;
            else
              weights(i) = sqrt(params.max_res_tol_sq * mu * (mu + 1) / residuals(i)) - mu;
            if (weights(i) < 0.0)   weights(i) = 0.0;
            if (weights(i) > 1.0)   weights(i) = 1.0;
    }
  }

  return weights;
}


 template <typename Scalar = double, typename... Args>
Scalar TLSUpdateMu(const Scalar& prev_mu, const GNCParams<Scalar> & params, Args &... args)
{

  return (prev_mu * params.gnc_factor);
}

} // end namespace of GM

}  // end namespace of gnso
