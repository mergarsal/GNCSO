#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <experimental/optional>
#include <Eigen/Dense>

// GNC
#include "gncso/Base/GNC.h"

#define SQRT2 1.41421356237
#define N_POINTS 100


typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, 3> Matrix;
typedef Eigen::Matrix<Scalar, 9, 9> Matrix9;
typedef Eigen::Matrix<Scalar, 3, N_POINTS> DataPoints;
typedef Eigen::Matrix<Scalar, 3, 1> Correspondence;
typedef Eigen::Matrix<Scalar, 3, 1> Vector;
typedef Eigen::Matrix<Scalar, 9, 1> Vector9;
typedef Eigen::Matrix<Scalar, 1, N_POINTS> Weights;



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


Matrix cross(const Vector & t)
{
        Matrix t_cross;
        t_cross.setZero();
        t_cross(0, 1) = -t(2); t_cross(0, 2) = t(1);
        t_cross(1, 0) = t(2); t_cross(1, 2) = -t(0);
        t_cross(2, 0) = -t(1); t_cross(2, 1) = t(0);
        return t_cross;
}


Matrix9 constructDataMatrix(const points_corr & bearing_vectors, const Weights & weights)
{
        Matrix9 C;
        Matrix9 temp;
        // clean output matrix
        C.setZero();

        for (int i = 0; i < bearing_vectors.src.cols(); i++)
        {
                // clean data
                temp.setZero();
                const Vector v0 = bearing_vectors.src.col(i);
                const Vector v1 = bearing_vectors.dst.col(i);
                const double weight = weights(i);
                for (int j = 0; j < 3; j++)    temp.block<3, 1>(j*3, 1) = v1[j] * v0;
                C += weight * temp * temp.transpose();
        }
        return 0.5 * (C + C.transpose());
}


int main() {

   std::cout << "Naive Test Essential Matrix\n";

   Scalar sigma_noise = 0.03;
   Scalar sigma_outliers = 20;  // add outliers as high noise
   // random seed
   std::srand(std::time(nullptr));

   DataPoints src_points, dst_points;
   points_corr points_str;

   // Create random pose
   Matrix R_ref = generateRandomRotationMatrix();
   Vector t_ref = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);

   // create the bearing vector
   // no FOV, no parallax
   for (size_t i = 0; i < src_points.cols(); ++i)
   {
     // create point in frame 0
     Correspondence p1 = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
     Correspondence p2 = R_ref.transpose() * (p1 - t_ref);

     src_points.col(i) = p1.normalized();
     dst_points.col(i) = p2.normalized();
   }


   // essential matrices
   Matrix E_est, E_ref;
   // compute ref essential matrix¡¡
   E_ref = cross(t_ref) * R_ref;
   E_ref = E_ref / (E_ref.norm()) * SQRT2;

   // save data into struct for optimization problem
   points_str.src = src_points;
   points_str.dst = dst_points;

   // Define the estimation function
   baseOpt::VariableEstimation<Matrix, Weights, points_corr> compute_svd_fcn = [](const Matrix& X, const Weights& weights, const points_corr& points)
  {
      Matrix9 C = constructDataMatrix(points, weights);

      Eigen::JacobiSVD<Matrix9> svd_C(C, Eigen::ComputeFullU | Eigen::ComputeFullV);

      // force the solution to be an essential matrix
      Vector9 e_null = svd_C.matrixV().col(8);
      Matrix E_null = Eigen::Map<Matrix>(e_null.data(), 3, 3);
      Eigen::JacobiSVD<Matrix> svd(E_null, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::DiagonalMatrix<double, 3> d(1, 1, 0);

      Matrix U = svd.matrixU();
      Matrix V = svd.matrixV();

      // force the solution to have SVD with rotation matrices
      if (U.determinant() < 0)    U.col(2) *= -1;
      if (V.determinant() < 0)    V.col(2) *= -1;

      return (U * d * (V).transpose());
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
     Weights residual;
     DataPoints p1 = points.src;
     residual.resize(p1.cols());

     for (size_t i = 0; i < p1.cols(); i++)
     {
       Vector v1 = (points.dst).col(i);
       Vector v0 = (points.src).col(i);
       residual(i) = (pow((v0.transpose() * X * v1), 2));
     }

     return (residual);
   };


   // Call solver w/t noise
   gncso::GNCParams<Scalar> options = gncso::GNCParams<Scalar>();
   options.inliers_threshold = 0.4;       // threshold to apply to the inliers. IF weight(i) > inliers_threshold, then 'i' is an inlier
   options.cost_diff_threshold = 0.0001;  // stop criteria. if the diff between two cost is lower than this value, stop
   options.max_res_tol_sq = 2e-06;  // maximum tolerance allowed (square)
   options.mu_threshold = 1+1e-08;          // for GM
   // Initial estimations

   Weights weights_initial;
   weights_initial.setOnes(1, N_POINTS);


   Matrix E_init = compute_svd_fcn(Matrix::Identity(), weights_initial, points_str);

   std::cout << "Solving problem without outliers nor noise...\n";
   
   // Solve the problem!!!
   gncso::GNCResult<Matrix, Weights, Scalar> results = gncso::GMGNC<Matrix, Weights, Scalar, points_corr>(compute_svd_fcn, compute_cost_fcn, compute_residuals_fcn, E_init, weights_initial, points_str, std::experimental::nullopt, std::experimental::nullopt, options);
   
   // extract solution
   E_est = results.x;
   std::cout << "Result without outliers\n------------\n";
   std::cout << "Ground truth essential matrix:\n" << E_ref << std::endl;
   std::cout << "Estimated essential matrix:\n" << E_est << std::endl;
   std::cout << "Number of detected inliers: " << (results.set_inliers).size() << std::endl;

   // Add noise

   for (size_t i = 0; i < N_POINTS; i++)
   {
       points_str.src(0, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       points_str.src(1, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       points_str.src(2, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;

       points_str.dst(0, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       points_str.dst(1, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       points_str.dst(2, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_noise;
       // normalize
       points_str.src.col(i).normalize();
       points_str.dst.col(i).normalize();
   }


   // Solve the problem!!!
   // GNC + GM
   std::cout << "Solving problem with noise...\n";
   
   gncso::GNCResult<Matrix, Weights, Scalar> results_noise = gncso::GMGNC<Matrix, Weights, Scalar, points_corr>(compute_svd_fcn, compute_cost_fcn, compute_residuals_fcn, E_init, weights_initial, points_str, std::experimental::nullopt, std::experimental::nullopt, options);
   
   // extract solution
   E_est = results_noise.x;
   std::cout << "Result with noise " << sigma_noise << "\n------------\n";
   std::cout << "Ground truth essential matrix:\n" << E_ref << std::endl;
   std::cout << "Estimated essential matrix:\n" << E_est << std::endl;
   std::cout << "Number of detected inliers: " << (results_noise.set_inliers).size() << std::endl;


  // Add outliers
  size_t n_outliers = 30;

  for (size_t i = 0; i < n_outliers; i++)
  {
    points_str.dst(0, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_outliers;
    points_str.dst(1, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_outliers;
    points_str.dst(2, i) += ((Scalar)std::rand() /  ((double) RAND_MAX) - 0.5) * 2.0 * sigma_outliers;
    points_str.dst.col(i).normalize();
  }

  // Solve the problem!!!
  std::cout << "Solving problem with outliers (GM)...\n";
  
  gncso::GNCResult<Matrix, Weights, Scalar> results_outliers = gncso::GMGNC<Matrix, Weights, Scalar, points_corr>(compute_svd_fcn, compute_cost_fcn, compute_residuals_fcn, E_init, weights_initial, points_str, std::experimental::nullopt, std::experimental::nullopt, options);
  
  // extract solution
  E_est = results_outliers.x;
  std::cout << "-----------GNC + GM --------------\n";
  std::cout << "Result with " << n_outliers << " outliers\n------------\n";
  std::cout << "Ground truth essential matrix:\n" << E_ref << std::endl;
  std::cout << "Estimated essential matrix:\n" << E_est << std::endl;
  std::cout << "Number of detected inliers: " << (results_outliers.set_inliers).size() << std::endl;

  


return 0;
}
