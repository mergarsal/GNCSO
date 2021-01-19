#pragma once

#include <functional>
#include <limits>
#include <vector>

#include "gncso/Base/BaseConcepts.h"
// Smooth
#include "gncso/Smooth/gnc_smooth.h"


// for the general fcns
using namespace baseOpt;


using namespace gncso; 


namespace refinements
{
  // TODO: include help

  template <typename Scalar = double>
  struct RefinParams : public GNCSmoothParams<Scalar>
  {
    /* max number of iterations */
    size_t max_iters;

    /**  Factor for the refinement  */
    Scalar factor_ref_residual;

    /* time limit for the refinement */
    Scalar max_computation_time;

    /* verbose */
    bool refin_verbose;

    /* constructor */
    RefinParams(size_t max_iters = 500, Scalar factor_ref_residual = 1.75,
      Scalar max_computation_time = 1000000, bool refin_verbose = 1): GNCSmoothParams<Scalar>(),
    max_iters(max_iters), factor_ref_residual(factor_ref_residual),
    max_computation_time(max_computation_time), refin_verbose(refin_verbose){};

  };  // end of struct RefinParams

/* status */
enum class RefinStatus
{
  CONVERGENCE_SET = 0,

  MAX_NR_ITERS,

  ELAPSED_TIME,

  USER_FUNCTION   // TODO: define this


};  // end of enum Status

// TODO: include struct for intermediate results

template <typename Variable, typename Weights, typename Scalar = double> struct RefinedResult
{
  RefinStatus refin_status;

  // Final items
  Variable x;

  /* final set of inliers */
  Weights set_inliers;

  /** The elapsed computation time */
  Scalar elapsed_time;

  /* number of iterations */
  Scalar nr_iterations;

  /* Upper bound for residuals */
  Scalar upper_bound_res;

  // TODO: include intermediate results

  RefinedResult() {};
};

        /* Variable estimation */
        // template <typename Variable, typename Weights, typename... Args>
        // using ComputeResidualsInliers = std::function<Weights(const Variable &X, const Weights& inliers, Args &... args)>;
        
        /* Compute max residual  */
        template <typename Variable, typename Weights, typename Scalar = double, typename... Args>
        using ComputeMaxCoefficient = std::function<Scalar(const Variable &X, const Weights& inliers, const Weights& residuals, Args &... args)>;
        

        /* Compute the set of inliers */
        template <typename Variable, typename Weights, typename Scalar = double, typename... Args>
        using ComputeSetInliers = std::function<Weights(const Variable &X, const Weights& inliers,
                  const Weights& residuals, const Scalar upper_bound_residuals, Args &... args)>;


};  // end of namespace refinements
