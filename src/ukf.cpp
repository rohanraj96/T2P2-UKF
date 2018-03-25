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

  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  previous_timestamp_ = 0.0;

  x_ = VectorXd(n_x_);
  // x_ << 1, 1, 1, 1, 1;

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

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_,0,
              0,std_laspy_*std_laspy_;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
  // cout << "PROCESS MEASUREMENT" << endl;

  if(!is_initialized)
  {

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];
      double v = sqrt(pow(rho_dot * cos(phi),2) + pow(rho_dot * sin(phi),2));
      x_ << rho * cos(phi), rho * sin(phi), v, 0, 0;
      // P_ << 0.2, 0, 0, 0, 0,
      //       0, 0.2, 0, 0, 0,
      //       0, 0, 1, 0, 0,
      //       0, 0, 0, 1, 0,
      //       0, 0, 0, 0, 1;
      previous_timestamp_ = meas_package.timestamp_;
    }

    else if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
      // P_ << 0.1, 0, 0, 0, 0,
      //       0, 0.1, 0, 0, 0,
      //       0, 0, 1, 0, 0,
      //       0, 0, 0, 1, 0,
      //       0, 0, 0, 0, 1;
      previous_timestamp_ = meas_package.timestamp_; 
    }

    is_initialized = true;
  }

  else
  {
    double del_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = meas_package.timestamp_;

    Prediction(del_t);

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
      UpdateRadar(meas_package);

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
  // x_aug_.fill(0.0);
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
  // cout << "OKAY" << endl;
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(5,5) = std_a_ * std_a_;
  P_aug_(6,6) = std_yawdd_ * std_yawdd_;

  int cols = 2 * n_aug_ + 1;
  
  MatrixXd Xsig_aug = GenerateSigmaPoints(x_aug_, P_aug_, lambda, cols);
  /**
  STEP 2: PREDICT SIGMA POINTS
  **/

  Xsig_pred = PredictSigmaPoints(Xsig_aug, delta_t, n_x_, cols, std_a_, std_yawdd_);

  /**
  STEP 3: PREDICT MEAN AND COVARIANCE
  **/

  x_ = Xsig_pred * weights_;

  P_.fill(0.0);
  for(int i = 0; i < cols; i++)
  {
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    x_diff(3) = angle_norm(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

MatrixXd UKF::GenerateSigmaPoints(VectorXd x, MatrixXd P, int lambda, int cols)
{

  int n = x.size();

  MatrixXd Xsig = MatrixXd(n, cols);
  MatrixXd A = P.llt().matrixL();
  Xsig.col(0) = x;

  double root = sqrt(lambda + n);

  for(int i = 0; i < n; i++)
  {
    Xsig.col( i + 1 ) = x + root * A.col(i);
    Xsig.col( i + 1 + n ) = x - root * A.col(i);
  }

  return Xsig;
}

MatrixXd UKF::PredictSigmaPoints(MatrixXd Xsig, double delta_t, int n_x, int cols, double nu_am, double nu_yawdd)
{
  MatrixXd Xsig_pred = MatrixXd(n_x, cols);
  //predict sigma points
  for (int i = 0; i < cols; i++)
  {
    //extract values for better readability
    double p_x = Xsig(0,i);
    double p_y = Xsig(1,i);
    double v = Xsig(2,i);
    double yaw = Xsig(3,i);
    double yawd = Xsig(4,i);
    double nu_a = Xsig(5,i);
    double nu_yawdd = Xsig(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  return Xsig_pred;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  STEP 1: PREDICT MEASUREMENT
  **/

  std::cout << "IN LIDAR" << endl;

  int z_dim = 2;

  MatrixXd Zsig = MatrixXd(z_dim, 2 * n_aug_ + 1);

  Zsig.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // double px = Xsig_pred(0, i);
    // double py = Xsig_pred(1, i);

    Zsig(0, i) = Xsig_pred(0, i);
    Zsig(1, i) = Xsig_pred(1, i);
  }

  // Predicted mean

  VectorXd z_pred = VectorXd(z_dim);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred += weights_(i) * Zsig.col(i);


  MatrixXd S_laser = MatrixXd(z_dim, z_dim);
  S_laser.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S_laser += weights_(i) * z_diff * (z_diff.transpose());
  }

  S_laser += R_lidar_;

  /**
  STEP 2: UPDATE STATE
  **/

  MatrixXd T_ = MatrixXd(n_x_, z_dim);
  T_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred.col(i) - x_;

    T_ += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = T_ * S_laser.inverse();
  VectorXd z = VectorXd(z_dim);
  z.fill(0.0);
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

  // cout << "IN RADAR" << endl;

  // int n_z = 3;
  // MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // for(int i = 0; i < 2 * n_aug_ + 1; i++)
  // {
  //   // cout << "IN FOR" << endl;
  //   double px = Xsig_pred(0, i);
  //   double py = Xsig_pred(1, i);
  //   double v = Xsig_pred(2, i);
  //   double w = Xsig_pred(3, i);

  //   double rho = sqrt(px * px + py * py);
  //   if(rho < 0.001)
  //     rho = 0.001;

  //   Zsig(0, i) = rho;
  //   Zsig(1, i) = atan2(py, px);
  //   Zsig(2, i) = (px * cos(w) * v + py * sin(w) * v)/rho;
  // }

  // VectorXd z_pred = VectorXd(n_z);
  // z_pred.fill(0.0);
  // for (int i=0; i < 2*n_aug_ + 1; i++) {
  //     z_pred = z_pred + weights_(i) * Zsig.col(i);
  //     // cout << "IN FOR" << endl;
  // }

  // MatrixXd S = MatrixXd(n_z,n_z);
  // S.fill(0.0);
  // for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
  //   //residual
  //   VectorXd z_diff = Zsig.col(i) - z_pred;

  //   //angle normalization
  //   while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  //   while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //   S = S + weights_(i) * z_diff * z_diff.transpose();
  // }

  // //add measurement noise covariance matrix
  // MatrixXd R = MatrixXd(n_z,n_z);
  // R <<    std_radr_*std_radr_, 0, 0,
  //         0, std_radphi_*std_radphi_, 0,
  //         0, 0,std_radrd_*std_radrd_;
  // S = S + R;

  // MatrixXd Tc = MatrixXd(n_x_, n_z);
  // Tc.fill(0.0);
  // cout << "okay" << endl;
  // for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

  //   //residual
  //   VectorXd z_diff = Zsig.col(i) - z_pred;
  //   //angle normalization
  //   while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  //   while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //   // state difference
  //   VectorXd x_diff = Xsig_pred.col(i) - x_;
  //   //angle normalization
  //   while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
  //   while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

  //   Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  // }

  // //Kalman gain K;
  // MatrixXd K = Tc * S.inverse();

  // //residual
  // VectorXd z(n_z);
  // z << meas_package.raw_measurements_[0],
  //       meas_package.raw_measurements_[1],
  //       meas_package.raw_measurements_[2];
  // VectorXd z_diff = z - z_pred;

  // //angle normalization
  // while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  // while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // //update state mean and covariance matrix
  // x_ = x_ + K * z_diff;
  // P_ = P_ - K*S*K.transpose();
  // cout << "okay" << endl;
  /**
  STEP 1: PREDICT MEASUREMENT
   **/

  std::cout << "IN RADAR" << endl;

  int z_dim = 3;

  MatrixXd Zsig = MatrixXd(z_dim, 2 * n_aug_ + 1);
  // MatrixXd R_radar(z_dim, z_dim);
  MatrixXd S_radar = MatrixXd(z_dim, z_dim);
  VectorXd z_pred = VectorXd(z_dim);

  Zsig.fill(0.0);
  S_radar.fill(0.0);
  z_pred.fill(0.0);

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = Xsig_pred(0, i);
    double py = Xsig_pred(1, i);
    double v = Xsig_pred(2, i);
    double w = Xsig_pred(3, i);

    double rho = sqrt(px*px + py*py);
    if(rho < 0.001)
      rho = 0.001;

    Zsig(0, i) = rho;
    Zsig(1, i) = atan2(py, px),
    Zsig(2, i) = (px*cos(w)*v+py*sin(w)*v)/rho;

  }

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred += weights_(i) * Zsig.col(i);


  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = angle_norm(z_diff(1));
    S_radar += weights_(i) * z_diff * (z_diff.transpose());
  }

  S_radar += R_radar_;

  /**
  STEP 2: UPDATE STATE
  **/

  MatrixXd T_ = MatrixXd(n_x_, z_dim);
  T_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = angle_norm(z_diff(1));
    VectorXd x_diff = Xsig_pred.col(i) - x_;

    T_ += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = T_ * S_radar.inverse();
  VectorXd z = VectorXd(z_dim);
  z.fill(0.0);
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
