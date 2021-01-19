#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <Eigen/Dense>

// GNC
#include "gncso/Base/GNC.h"
#include "gncso/Base/BaseConcepts.h"

typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, 3> Matrix;
typedef Eigen::Matrix<Scalar, 3, 10> DataPoints;
typedef Eigen::Matrix<Scalar, 1, 10> Weights;

using namespace std;

struct points_corr
{
  DataPoints src;
  DataPoints dst;
};

Matrix generateRandomRotationMatrix(void)
{
  Matrix R_aux, R_ref = Matrix::Identity();

  // Euler angles
  Scalar theta = ((Scalar)std::rand() - 0.5) * 2.0 * 5.0;

  // Rotation around X-axis
  R_aux.setZero();
  R_aux << 1, 0, 0, 0, cos(theta), -sin(theta), 0, sin(theta), cos(theta);
  R_ref *= R_aux;

  // Rotation around Y-axis
  R_aux.setZero();
  R_aux << cos(theta), 0, sin(theta), 0, 1, 0, -sin(theta),0, cos(theta);

  // update ref. rotation
  R_ref *= R_aux;

  // Rotation around Z-axis
  R_aux.setZero();
  R_aux << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;

  // update ref. rotation
  R_ref *= R_aux;

  return R_ref;
}

Scalar computeGeodesicDistance(Matrix R1, Matrix R2)
{
  return (acos((((R1.transpose() * R2).trace() - 1 ) * 0.5)));
}

int main() {

   std::cout << "Test Rotation\n";

   Scalar sigma_noise = 0.03;
   Scalar sigma_outliers = 20;  // add outliers as high noise
   // random seed
   std::srand(std::time(nullptr));

   DataPoints src_points, dst_points;
   points_corr points_str;

   for (size_t i = 0; i < src_points.cols(); ++i) {
     src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
   }



   // rotation matrices
   Matrix R_est, R_ref;

   // generate a random matrix
   R_ref = generateRandomRotationMatrix();

   // compute the points wrt the second frame
   dst_points = R_ref * src_points;

   // save data into struct for optimization problem
   points_str.src = src_points;
   points_str.dst = dst_points;

   // Define the estimation function
   baseOpt::VariableEstimation<Matrix, Weights, points_corr> compute_svd_fcn = [](const Matrix& X, const Weights& weights, const points_corr& points)
  {
      DataPoints src = points.src;
      DataPoints dst = points.dst;
      Matrix corr_matrix = src * weights.asDiagonal() * dst.transpose();


      Eigen::JacobiSVD<Eigen::Matrix3d> svd(corr_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3d U = svd.matrixU();
      Eigen::Matrix3d V = svd.matrixV();

      // force the solution to be a rotation matrix
      if (U.determinant() * V.determinant() < 0)    V.col(2) *= -1;

      Matrix temp = V * U.transpose();
      return temp;
   };

   // Define the cost function

   baseOpt::ComputeCost<Matrix, Weights, Scalar, points_corr> compute_cost_fcn = [](const Matrix& X, const Weights& weights, const Weights& residuals_sq, const points_corr& points )
   {
    Scalar cost = 0;
    size_t N = weights.cols();
    for (size_t i = 0; i < N; i ++)    cost += weights(i) * residuals_sq(i);
    return (cost);
  };

   // Define how to compute the residuals ^2
   baseOpt::ComputeResiduals<Matrix, Weights, points_corr> compute_residuals_fcn = [](const Matrix& X, const points_corr& points)
   {
     DataPoints src = points.src;
     DataPoints dst = points.dst;
     DataPoints diffs = (dst - X * src).array().square();
     Weights res_sq = diffs.colwise().sum();
     return (res_sq);
   };


   // Call solver w/t noise
   gncso::GNCParams<Scalar> options = gncso::GNCParams<Scalar>();
   options.inliers_threshold = 0.4;  // threshold to apply to the inliers. IF weight(i) > inliers_threshold, then 'i' is an inlier
   options.cost_diff_threshold = 0.0001;  // stop criteria. if the diff between two cost is lower than this value, stop
   options.max_res_tol_sq = 0.004;  // maximum tolerance allowed (square)

   // Initial estimations
   Matrix R_init = Matrix::Identity();
   Weights weights_initial;
   weights_initial.setOnes(1, 10);
   // Solve the problem!!!
   gncso::GNCResult<Matrix, Weights, Scalar> results = gncso::GMGNC<Matrix, Weights, Scalar, points_corr>(compute_svd_fcn, compute_cost_fcn, 
                                                compute_residuals_fcn, R_init, weights_initial, points_str, 
                                                std::experimental::nullopt, std::experimental::nullopt, options);
   // extract solution
   R_est = results.x;
   std::cout << "Result without outliers\n------------\n";
   std::cout << "Ground truth rotation matrix:\n" << R_ref << std::endl;
   std::cout << "Estimated rotation matrix:\n" << R_est << std::endl;
   std::cout << "Geodesic distance between both rotations: " << computeGeodesicDistance(R_ref, R_est) << std::endl;
   std::cout << "Detected inliers: " << results.set_inliers << std::endl;

   // Add noise
   size_t N = 10;

   for (size_t i = 0; i < N; i++)
   {
       points_str.src(0, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       points_str.src(1, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       points_str.src(2, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;

       points_str.dst(0, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       points_str.dst(1, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       points_str.dst(2, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
   }

   // Solve the problem!!!
   // GNC + GM
   gncso::GNCResult<Matrix, Weights, Scalar> results_noise = gncso::GMGNC<Matrix, Weights, Scalar, points_corr>(compute_svd_fcn, 
                                compute_cost_fcn, compute_residuals_fcn, R_init, weights_initial, points_str, 
                                std::experimental::nullopt, std::experimental::nullopt, options);
   // extract solution
   R_est = results_noise.x;
   std::cout << "Result with noise " << sigma_noise << "\n------------\n";
   std::cout << "Ground truth rotation matrix:\n" << R_ref << std::endl;
   std::cout << "Estimated rotation matrix:\n" << R_est << std::endl;
   std::cout << "Geodesic distance between both rotations: " << computeGeodesicDistance(R_ref, R_est) << std::endl;
   std::cout << "Detected inliers: " << results_noise.set_inliers << std::endl;

  // Add outliers
  size_t n_outliers = 4;

  for (size_t i = 0; i < n_outliers; i++)
  {
    points_str.dst(0, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_outliers;
    points_str.dst(1, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_outliers;
    points_str.dst(2, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_outliers;
  }

  // Solve the problem!!!
  gncso::GNCResult<Matrix, Weights, Scalar> results_outliers = gncso::GMGNC<Matrix, Weights, Scalar, points_corr>(compute_svd_fcn, compute_cost_fcn, 
                                                compute_residuals_fcn, R_init, weights_initial, points_str, 
                                                std::experimental::nullopt, std::experimental::nullopt, options);
  // extract solution
  R_est = results_outliers.x;
  std::cout << "-----------GNC + GM --------------\n";
  std::cout << "Result with " << n_outliers << " outliers\n------------\n";
  std::cout << "Ground truth rotation matrix:\n" << R_ref << std::endl;
  std::cout << "Estimated rotation matrix:\n" << R_est << std::endl;
  std::cout << "Geodesic distance between both rotations: " << computeGeodesicDistance(R_ref, R_est) << std::endl;
  std::cout << "Detected inliers: " << results_outliers.set_inliers << std::endl;
  std::cout << "Final weights: " << results_outliers.weights << std::endl;

  


return 0;
}
