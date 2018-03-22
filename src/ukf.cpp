#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

float angle_norm(double angle);

UKF::UKF() {

  is_initialized = false;

  use_laser_ = true;

  use_radar_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  previous_timestamp_ = 0.0;

  x_ = VectorXd(n_x_);
  x_ << 1, 1, 1, 1, 1;

  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0 ,0, 1;

  Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred.fill(0.0);
  std_a_ = 1;
  std_yawdd_ = 1;
  std_laspx_ = 0.15;
  std_laspy_ = 0.15;
  std_radr_ = 0.3;
  std_radphi_ = 0.03;
  std_radrd_ = 0.3;

  lambda = 3 - n_x_;

  weights_ = VectorXd(2 * n_aug_ + 1);
  double first = lambda / (lambda + n_aug_);
  weights_(0) = first;
  for(int i = 1; i < 2 * n_aug_ + 1; i++)
    weights_(i) = 0.5 / (lambda + n_aug_);

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
  cout << "PROCESS MEASUREMENT" << endl;

  if(!is_initialized)
  {

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
      P_ << 0.2, 0, 0, 0, 0,
            0, 0.2, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
      previous_timestamp_ = meas_package.timestamp_;
    }

    else if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
      P_ << 0.1, 0, 0, 0, 0,
            0, 0.1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
      previous_timestamp_ = meas_package.timestamp_; 
    }

    is_initialized = true;
  }

  else
  {
    float del_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = meas_package.timestamp_;

    Prediction(del_t);
    cout << "OKAY4" << endl;

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
      UpdateRadar(meas_package);

    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_)
      cout << "WE HAVE RADAR" << endl;

    else if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
      cout << "OKAY5" << endl;
      UpdateLidar(meas_package);
    }
  }
}

float angle_norm(double phi)
{
  float tan_theta = tan(phi);
  return atan(tan_theta);
}

void UKF::Prediction(double delta_t) {

  /**
  STEP 1: GENERATE SIGMA POINTS
  **/

  cout << "IN PREDICT" << endl;

  VectorXd x_aug_(n_aug_);
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
  // cout << "OKAY" << endl;
  MatrixXd P_aug_(n_aug_, n_aug_);
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(5,5) = std_a_ * std_a_;
  P_aug_(6,6) = std_yawdd_ * std_yawdd_;
  double lambda_aug = 3 - n_aug_;
  // cout << "OKAY" << endl;
  double multiplier = sqrt(lambda_aug + n_aug_);
  MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
  MatrixXd root = P_aug_.llt().matrixL();
  // cout << "OKAY" << endl;
  Xsig_aug.col(0) = x_aug_;

  for(int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug_ + multiplier * root.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug_ - multiplier * root.col(i);
    // cout << "OKAY" << endl;
  }
  cout << "OKAY1" << endl;

  /**
  STEP 2: PREDICT SIGMA POINTS
  **/

  int cols = 2 * n_aug_ + 1;
  
  MatrixXd x_k(n_x_, cols);
  MatrixXd x_change(n_x_, cols);
  MatrixXd noise(n_x_, cols);
  // cout << "OKAY" << endl;

  for(int i = 0; i < n_x_; i++)
    for(int j = 0; j < cols; j++)
      x_k(i, j) = Xsig_aug(i, j);

  for(int i = 0; i < cols; i++)
  {
    // cout << "OKAY" << endl;

    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double psi = Xsig_aug(3,i);
    double psi_dot = Xsig_aug(4,i);
    double noise_lin = Xsig_aug(5,i);
    double noise_rad = Xsig_aug(6,i);

    if(psi_dot > 0.001)
    {
      x_change(0,i) = (v/psi_dot)*(sin(psi + (psi_dot*delta_t)) - sin(psi));
      x_change(1,i) = (v/psi_dot)*(-cos(psi + (psi_dot*delta_t)) + cos(psi));
      x_change(2,i) = 0;
      x_change(3,i) = psi_dot * delta_t;
      x_change(4,i) = 0;
    }

    else
    {
      x_change(0,i) = v * cos(psi) * delta_t;
      x_change(1,i) = v * sin(psi) * delta_t;
      x_change(2,i) = 0;
      x_change(3,i) = psi_dot * delta_t;
      x_change(4,i) = 0;
            
    }
        
    noise(0,i) = 0.5 * delta_t * delta_t * cos(psi) * noise_lin;
    noise(1,i) = 0.5 * delta_t * delta_t * sin(psi) * noise_lin;
    noise(2,i) = delta_t * noise_lin;
    noise(3,i) = 0.5 * delta_t * delta_t * noise_rad;
    noise(4,i) = delta_t * noise_rad;

    }
  
  // cout << "OKAY PREV" << endl;
  Xsig_pred = x_k + x_change + noise;
  cout << "OKAY2" << endl;  
  /**
  STEP 3: PREDICT MEAN AND COVARIANCE
  **/

  // cout << "OKAY" << endl;
  for(int i = 0; i < cols; i++)
  {
    x_ += weights_(i) * Xsig_pred.col(i);
    x_(3) = angle_norm(x_(3));
  }

  for(int i = 0; i < cols; i++)
  {
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    P_ += weights_(i) * (x_diff) * (x_diff.transpose()); 
  }
  cout << "OKAY3" << endl;

}
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  STEP 1: PREDICT MEASUREMENT
  **/

  std::cout << "IN LIDAR" << endl;

  int z_dim = 2;

  MatrixXd Zsig(z_dim, 2 * n_aug_ + 1);
  MatrixXd R_laser(z_dim, z_dim);
  MatrixXd S_laser(z_dim, z_dim);
  VectorXd z_pred(z_dim);

  R_laser << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = Xsig_pred(0, i);
    double py = Xsig_pred(1, i);

    Zsig.col(i) << px, py;
  }

  // Predicted mean

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred += weights_(i) * Zsig.col(i);

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S_laser += weights_(i) * z_diff * (z_diff.transpose());
  }

  S_laser += R_laser;

  /**
  STEP 2: UPDATE STATE
  **/

  MatrixXd T_(n_x_, z_dim);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred.col(i) - x_;

    T_ += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = T_ * S_laser.inverse();
  VectorXd z(z_dim);
  z << meas_package.raw_measurements_[0],
           meas_package.raw_measurements_[1];

  VectorXd z_diff = z - z_pred;

  MatrixXd NIS = (z_diff.transpose()) * S_laser.inverse() * z_diff;
  x_ += K*z_diff;
  P_ -= K*S_laser*K.transpose();

  cout << "NIS: " << endl;
  cout << NIS;
  cout << endl << endl;

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  STEP 1: PREDICT MEASUREMENT
   **/

  std::cout << "IN RADAR" << endl;

  int z_dim = 3;

  MatrixXd Zsig(z_dim, 2 * n_aug_ + 1);
  MatrixXd R_radar(z_dim, z_dim);
  MatrixXd S_radar(z_dim, z_dim);
  VectorXd z_pred(z_dim);

  R_radar << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = Xsig_pred(0, i);
    double py = Xsig_pred(1, i);
    double v = Xsig_pred(2, i);
    double w = Xsig_pred(3, i);

    double rho = sqrt(px*px + py*py);
    if(rho < 0.001)
      rho = 0.001;
    Zsig.col(i) << rho, atan2(py, px), (px*cos(w)*v+py*sin(w)*v)/rho;

  }

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred += weights_(i) * Zsig.col(i);


  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = angle_norm(z_diff(1));
    S_radar += weights_(i) * z_diff * (z_diff.transpose());
  }

  S_radar += R_radar;

  /**
  STEP 2: UPDATE STATE
  **/

  MatrixXd T_(n_x_, z_dim);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = angle_norm(z_diff(1));
    VectorXd x_diff = Xsig_pred.col(i) - x_;

    T_ += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = T_ * S_radar.inverse();
  VectorXd z(z_dim);
  z << meas_package.raw_measurements_[0],
           meas_package.raw_measurements_[1],
            meas_package.raw_measurements_[2];

  VectorXd z_diff = z - z_pred;
  z_diff(1) = angle_norm(z_diff(1));

  MatrixXd NIS = (z_diff.transpose()) * S_radar.inverse() * z_diff;

  x_ += K*z_diff;
  P_ -= K*S_radar*K.transpose();

  cout << "NIS: " << endl;
  cout << NIS;
  cout << endl << endl;
}
