#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  // The estimation vector size should not be zero and should equal the ground
  // truth vector size
  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    return rmse;
  }
  // Accumulate squared residuals
  for (int i = 0; i < estimations.size(); i++) {
    VectorXd c = estimations[i] - ground_truth[i];
    c = c.array() * c.array();
    rmse += c;
  }
  rmse = rmse.array() / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  // Recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  // Precompute denominators
  float sum = px*px + py*py;
  float dist = sqrt(sum);
  float cube = sum * dist;
  // Check division by zero
  if (fabs(sum) < 0.001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }
  // Compute the Jacobian matrix
  Hj << px/dist, py/dist, 0, 0,
       -py/sum, px/sum, 0, 0,
       py*(vx*py-vy*px)/cube, px*(vy*px-vx*py)/cube, px/dist, py/dist;
  return Hj;
}
