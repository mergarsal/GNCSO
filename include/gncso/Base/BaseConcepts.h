#pragma once

#include <vector>

/* These functions are general and do not depend on GNC */
namespace baseOpt 
{

        /* Variable estimation */
        template <typename Variable, typename Weights, typename... Args>
        using VariableEstimation = std::function<Variable(const Variable &X, const Weights& weights, Args &... args)>;

        /* Compute residuals */
        template <typename Variable, typename Weights, typename... Args>
        using ComputeResiduals = std::function<Weights(const Variable &X, Args &... args)>;

        /* Compute cost */
        template <typename Variable, typename Weights, typename Scalar = double, typename... Args>
        using ComputeCost = std::function<Scalar(const Variable &X, const Weights& weights, const Weights& residuals, Args &... args)>;

        
        /* Update weights in the problem*/
        template <typename Variable, typename Scalar, typename Weights, typename... Args>
        using UpdateParamsInProblem = std::function<void(const Variable &X, const Weights& new_weights, Args &... args)>;

        /* Provide an alternative initialization */
        template <typename Variable, typename Scalar, typename Weights, typename... Args>
        using InitializeVariableX = std::function<Variable(const Variable &X, const Weights& weights, Args &... args)>;



};  // end of namespace baseOpt
