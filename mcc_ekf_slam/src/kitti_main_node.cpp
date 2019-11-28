#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
 #include <tf2/LinearMath/Quaternion.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <sstream>
#include "Eigen/Dense"
#include <fstream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include<ctime>
#include<chrono>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace std::chrono;

enum class DataPointType{
  LIDAR, STEREO, RADAR,STATE
};

VectorXd convert_polar_to_cartesian(const VectorXd& v){

  VectorXd cartesian_vec(4);

  const double rho = v(0);
  const double phi = v(1);
  const double drho = v(2);

  const double px = rho * cos(phi);
  const double py = rho * sin(phi);
  const double vx = drho * cos(phi);
  const double vy = drho * sin(phi);

  cartesian_vec << px, py, vx, vy;
  return cartesian_vec;
}
VectorXd convert_cartesian_to_polar(const VectorXd& v){

  const double THRESH = 0.0001;
  VectorXd polar_vec(3);

  const double px = v(0);
  const double py = v(1);
  const double vx = v(2);
  const double vy = v(3);

  const double rho = sqrt( px * px + py * py);
  const double phi = atan2(py, px); //accounts for atan2(0, 0)
  const double drho = (rho > THRESH) ? ( px * vx + py * vy ) / rho : 0.0;

  polar_vec << rho, phi, drho;
  return polar_vec;
}



class DataPoint{

  private:
    long long timestamp;
    bool initialized;
    DataPointType data_type;
    VectorXd raw;

  public:
    DataPoint();
    DataPoint(const long long timestamp, const DataPointType data_type, const VectorXd raw);
    void set(long timestamp, const DataPointType data_type, const VectorXd raw);
    VectorXd get() const;
    VectorXd get_state() const;
    DataPointType get_type() const;
    long long get_timestamp() const;
    void print() const;
};

inline DataPoint::DataPoint(){
  this->initialized = false;
}

inline DataPoint::DataPoint(const long long timestamp, const DataPointType data_type, const VectorXd raw){
  this->set(timestamp, data_type, raw);
}


inline void DataPoint::set(const long timestamp, const DataPointType data_type, const VectorXd raw){
  this->timestamp = timestamp;
  this->data_type = data_type;
  this->raw = raw;
  this->initialized = true;
}

inline VectorXd DataPoint::get() const{
  return this->raw;
}

inline VectorXd DataPoint::get_state() const{

  VectorXd state(6);

  if ((this->data_type == DataPointType::LIDAR)||(this->data_type == DataPointType::STEREO)){

    //cout<<"get_state="<<this->raw<<"\n";
    double x = this->raw(0);
    double y = this->raw(1);
    double z = this->raw(2);
    state << x,y,z,0.0,0.0,0.0;

  } else if (this->data_type == DataPointType::RADAR){

    state = convert_polar_to_cartesian(this->raw);

  } else if (this->data_type == DataPointType::STATE){

    state = this->raw;
  }

  return state;
}

inline long long DataPoint::get_timestamp() const{
  return this->timestamp;
}

inline DataPointType DataPoint::get_type() const{
  return this->data_type;
}

inline void DataPoint::print() const{

  if (this->initialized){

    cout << "Timestamp: " << this->timestamp << endl;
    cout << "Sensor ID: " << static_cast<int>(this->data_type) << " (LIDAR = 0 | RADAR = 1 | STATE = 2)" << endl;
    cout << "Raw Data: " << endl;
    cout << this->raw << endl;

  } else {

    cout << "DataPoint is not initialized." << endl;
  }
}

class KalmanFilter{

  private:
    int n;
    VectorXd x;
    MatrixXd P;
    MatrixXd F;
    MatrixXd Q;
    MatrixXd I;
    MatrixXd MCC_KF;
    ofstream param_vals;


  public:
    int cntr=0;
    double sigmakf_val;
    KalmanFilter()
    {
        param_vals.open("param_vals.txt");
    };
    void start(const int nin, const VectorXd& xin, const MatrixXd& Pin, const MatrixXd& Fin, const MatrixXd& Qin);
    void setQ(const MatrixXd& Qin);
    void updateF(const double dt);
    VectorXd get() const;
    void predict();
    void update(const VectorXd& z, const MatrixXd& H, const VectorXd& Hx, const MatrixXd& R,int flag);
};


class FusionEKF{

  private:
    const int n = 6;
    const int lidar_n = 3;
    const int stereo_n = 3;
    const int radar_n = 3;
    const double ax = 9.0; //5.0
    const double ay = 9.0; //5.0
    const double az = 9.0;
    bool initialized;
    long long timestamp;
    MatrixXd P;
    MatrixXd F;
    MatrixXd Q;
    MatrixXd radar_R;
    MatrixXd lidar_R;
    MatrixXd stereo_R;
    MatrixXd lidar_H;
    MatrixXd stereo_H;
    MatrixXd MCC_KF;
    KalmanFilter KF;
    int fl=0;
  public:
  	int cntr=0;
  	double sigma_val;
    FusionEKF();
    void updateQ(const double dt,const MatrixXd& u_Q);
    void compute(const DataPoint& data,const MatrixXd& u_Q,int flag);
    void start(const DataPoint& data,const MatrixXd& u_Q);
    void process(const DataPoint& data,const MatrixXd& u_Q,int flag);
    VectorXd get() const;
};

inline FusionEKF::FusionEKF(){

  this->initialized = false;

  this->lidar_R = MatrixXd(this->lidar_n, this->lidar_n);
  this->stereo_R = MatrixXd(this->lidar_n, this->lidar_n);
  this->radar_R = MatrixXd(this->radar_n, this->radar_n);
  this->lidar_H = MatrixXd(this->lidar_n, this->n);
  this->stereo_H = MatrixXd(this->lidar_n, this->n);

  this->P = MatrixXd(this->n, this->n);
  this->F = MatrixXd::Identity(this->n, this->n);
  this->Q = MatrixXd::Zero(this->n, this->n);
  this->MCC_KF=MatrixXd(this->n,this->n);

  this->lidar_R << 0.0225, 0.0,0.0,
                   0.0, 0.0225,0.0,
                   0.0,0.0,0.0225;
  

  this->stereo_R << 0.0225, 0.0,0.0,
                   0.0, 0.0225,0.0,
                   0.0,0.0,0.0225;               

  this->radar_R  << 0.09, 0.0, 0.0,
                    0.0, 0.0009, 0,
                    0.0, 0.0, 0.09;

  this->lidar_H << 1.0, 0.0, 0.0, 0.0,0.0,0.0,
                   0.0, 1.0, 0.0, 0.0,0.0,0.0,
                   0.0, 0.0, 1.0, 0.0,0.0,0.0;

  this->stereo_H <<1.0, 0.0, 0.0, 0.0,0.0,0.0,
                   0.0, 1.0, 0.0, 0.0,0.0,0.0,
                   0.0, 0.0, 1.0, 0.0,0.0,0.0;            

  this->P << 0.1, 0.0, 0.0, 0.0,0.0,0.0,
             0.0, 0.1, 0.0, 0.0,0.0,0.0,
             0.0, 0.0, 1000.0, 0.0,0.0,0.0,
             0.0, 0.0, 0.0, 1000,0.0,0.0,
             0.0, 0.0, 0.0,0.0,1000.0,0.0,
              0.0, 0.0, 0.0,0.0,0.0,1000.0;

  
  
  
}

inline void FusionEKF::updateQ(const double dt,const MatrixXd& u_Q){

  // const double dt2 = dt * dt;
  // const double dt3 = dt * dt2;
  // const double dt4 = dt * dt3;
  // const double dt5 = dt * dt4;
  // const double dt6 = dt * dt5;

  // const double r11 = dt6 * this->ax / 6;
  // const double r13 = dt4 * this->ax / 4;
  // const double r15 = dt2 * this->ax / 2;
  
  // const double r22 = dt6 * this->ay / 6;
  // const double r24 = dt4 * this->ay / 4;
  // const double r26 = dt2 * this->ay / 2;


  // const double r31 = dt3 * this->ax / 2;
  // const double r33 = dt2 * this->ax;
  // const double r33 = dt2 * this->ax;
  
  // const double r42 = dt3 * this->ay / 2;
  // const double r44 = dt2 * this->ay;
  
  // this->Q << r11, 0.0, r13, 0.0, r15, 0.0,
  //            0.0, r22, 0.0, r24, 0.0, r26,
  //            r31, 0.0, r33, 0.0, r35, 0.0,
  //            0.0, r42, 0.0, r44, 0.0, r46,
  //            r51, 0.0, r53, 0.0, r55, 0.0,
  //            0.0, r62, 0.0, r64, 0.0, r66;
  
  this->KF.setQ(u_Q);
}

MatrixXd calculate_jacobian(const VectorXd &v){

  const double THRESH = 0.0001;
  MatrixXd H = MatrixXd::Zero(3, 4);

  const double px = v(0);
  const double py = v(1);
  const double vx = v(2);
  const double vy = v(3);

  const double d_squared = px * px + py * py;
  const double d = sqrt(d_squared);
  const double d_cubed = d_squared * d;

  if (d >= THRESH){

    const double r11 = px / d;
    const double r12 = py / d;
    const double r21 = -py / d_squared;
    const double r22 = px / d_squared;
    const double r31 = py * (vx * py - vy * px) / d_cubed;
    const double r32 = px * (vy * px - vx * py) / d_cubed;

    H << r11, r12, 0.0, 0.0,
         r21, r22, 0.0, 0.0,
         r31, r31, r11, r12;
  }

  return H;
}


inline void FusionEKF::start(const DataPoint& data,const MatrixXd& u_Q){

  this->timestamp = data.get_timestamp();
  VectorXd x = data.get_state();
 // cout<<"x="<<x<<endl;
  this->KF.start(this->n, x, this->P, this->F, this->Q);
  this->KF.sigmakf_val=this->sigma_val;
  this->initialized = true;
}

inline void FusionEKF::compute(const DataPoint& data,const MatrixXd& u_Q,int flag){

  /**************************************************************************
   * PREDICTION STEP
   **************************************************************************/
  const double dt = (data.get_timestamp() - this->timestamp) / 1.e6;
  this->timestamp = data.get_timestamp();

  this->updateQ(dt,u_Q);
  this->KF.updateF(dt);
  this->KF.predict();

  /**************************************************************************
   * UPDATE STEP
   **************************************************************************/
  const VectorXd z = data.get();
  const VectorXd x = this->KF.get();

  VectorXd Hx;
  MatrixXd R;
  MatrixXd H;

  if (data.get_type() == DataPointType::RADAR){

    VectorXd s = data.get_state();
    H = calculate_jacobian(s);
    Hx = convert_cartesian_to_polar(x);
    R =  this->radar_R;

  } else if ((data.get_type() == DataPointType::LIDAR) || (data.get_type() == DataPointType::STEREO) )
  {

    H = this->lidar_H;
    Hx = this->lidar_H * x;
    R = this->lidar_R;
  }

  this->KF.update(z, H, Hx, R,flag);
}

inline void FusionEKF::process(const DataPoint& data,const MatrixXd& u_Q,int flag){
	//cout<<"process="<<u_Q<<endl;
 this->initialized ? this->compute(data,u_Q,flag) : this->start(data,u_Q);
}

inline VectorXd FusionEKF::get() const{
  return this->KF.get();
}



void KalmanFilter::start(
  const int nin, const VectorXd& xin, const MatrixXd& Pin, const MatrixXd& Fin, const MatrixXd& Qin){

  this->n = nin;
  this->I = MatrixXd::Identity(this->n, this->n);
  this->x = xin;
  this->P = Pin;
  this->F = Fin;
  this->Q = Qin;
  //this->MCC_KF=MatrixXd(this->n,this->n);
  //this->MCC_KF << 4.0, 0.0, 0.0, 0.0,
            // 0.0, 4.0, 0.0, 0.0,
             //0.0, 0.0, 3.0, 0.0,
             ///0.0, 0.0, 0.0, 3.0; 
}

inline void KalmanFilter::setQ(const MatrixXd& Qin){
  this->Q = Qin;
}

inline void KalmanFilter::updateF(const double dt){
  this->F(0, 3) = dt;
  this->F(1, 4) = dt;
  this->F(2, 5) = dt;
}

inline VectorXd KalmanFilter::get() const{
  return this->x;
}

inline void KalmanFilter::predict(){
  this->x = this->F * this->x;
  this->P = (this->F * this->P * this->F.transpose() )+ this->Q;
}

inline void KalmanFilter::update(const VectorXd& z, const MatrixXd& H, const VectorXd& Hx, const MatrixXd& R,int flag){
 // std::cout<<"\nctr="<<cntr<< "-----------------------------------------------------------";
  //Added for MCC-KF
  MatrixXd Iden=MatrixXd::Identity(this->n,this->n);
  VectorXd y = z - Hx;
  this->param_vals<<z.transpose()<<"   ";
  this->param_vals<<y.transpose()<<"   ";

  //std::cout<<"\n z= ---------->"<<z;
  //std::cout<<"\n y=Z-hx---------->"<<y;

  MatrixXd inverse_R=R.inverse();
  MatrixXd inverse_P=this->P.inverse();
  //VectorXd innov=z-Hx;
  VectorXd innov_x=this->x-(this->F*this->x);

  //std::cout<<"\ninnov_x= "<<innov_x;
  //std::cout<< "\ninnov = "<<innov;
  MatrixXd cov_mcc_kf=this->MCC_KF;
  double norm_innov=sqrt(y.transpose()*inverse_R*y);

  double norm_innov_d=sqrt(innov_x.transpose()*inverse_P*innov_x);

  //std::cout<<"\nnorm_innov = "<<norm_innov;
  //double norm_innov_q=sqrt()
  double sigma=10;
  //std::cout<<"\nsigma= "<<sigma;
  double K1=exp(-(norm_innov*norm_innov)/(2*sigma*sigma));
  double K2=exp(-(norm_innov_d*norm_innov_d)/(2*sigma*sigma));
  //std::cout<<"\nk1/k2= "<<K1/K2;

  double L1;
  if (flag==0)
    L1=1;
  else if(flag==1)
    L1=K1;
  this->param_vals<<L1<<"  ";
  //std::cout<<"\n y=Z-hx---------->"<<y;
  std::cout<<"\nL1= "<<L1;
  //MatrixXd pre_K=cov_mcc_kf.inverse()+((K1)*H.transpose()*inverse_R*H);
  //MatrixXd gain=pre_K.inverse()*(K1)*H.transpose()*inverse_R;
  
  //if (y.size() == 3) y(1) = atan2(sin(y(1)), cos(y(1))); //if radar measurement, normalize angle

  //MatrixXd PHt = this->P * H.transpose();
  //MatrixXd S = H * PHt + R;
  MatrixXd K = (this->P.inverse() + ((L1)*H.transpose()*inverse_R*H)).inverse()*(L1)*H.transpose()*inverse_R;
  this->x = this->x + (K * y);
  this->param_vals<<this->x.transpose()<<"\n";
  MatrixXd val1=(this->I - (K * H));
  this->P= (val1 * this->P * val1.transpose()) + (K*R*K.transpose());
}



nav_msgs::Path lidar_path,stereo_path,gt_path,mcc_path,ekf_path;
ros::Publisher lidar_path_pub;
ros::Publisher stereo_path_pub;
ros::Publisher gt_path_pub;
ros::Publisher mcc_slam_pub,ekf_slam_pub;
FusionEKF ekf,mcc;
DataPoint stereo_d,lidar_d,lidar_dna;
ros::Publisher mcc_odom_pub,lidar_odom_pub,ekf_odom_pub,noattack_pub;
int ctr=0;
void LidarOdomToPath(const nav_msgs::Odometry::ConstPtr& msg)
{
  // ROS_INFO("Seq: [%d]", msg->header.seq);
  // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
  // ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  // ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);
  ctr++;
  cout<<"ctr="<<ctr<<endl;
  geometry_msgs::PoseStamped data;
  data.pose.position.x=msg->pose.pose.position.x;
  data.pose.position.y=msg->pose.pose.position.y;
  data.pose.position.z=msg->pose.pose.position.z;
  data.pose.orientation.x=msg->pose.pose.orientation.x;
  data.pose.orientation.y=msg->pose.pose.orientation.y;
  data.pose.orientation.z=msg->pose.pose.orientation.z;
  data.pose.orientation.w=msg->pose.pose.orientation.w;
  
  long int timestamp=duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
 // cout<<timestamp<<endl;
 // DataPoint lidar_d;

  
  //mcc_odom.twist.twist.linear.x = prediction(3);
  //mcc_odom.twist.twist.linear.y = prediction(4);
  //mcc_odom.twist.twist.angular.z = prediction(5);
  
  VectorXd d_vals(3),dna_vals(3);
  dna_vals<<data.pose.position.x,data.pose.position.y,data.pose.position.z;
  float attx=15.0;
  float atty=15.0;

  if((ctr==50 || ctr==60)||(ctr==75 || ctr==88))
    {   
    	//data.pose.position.x=data.pose.position.x+attx;
    	//data.pose.position.y=data.pose.position.y+atty;

    	d_vals<<data.pose.position.x+attx,data.pose.position.y+atty,data.pose.position.z;}
  else
    {d_vals<<data.pose.position.x,data.pose.position.y,data.pose.position.z;}
  //cout<<d_vals<<endl;
  
  lidar_d.set(timestamp,DataPointType::LIDAR,d_vals);
  lidar_dna.set(timestamp,DataPointType::LIDAR,dna_vals);
  MatrixXd Q(6,6);
  int k=0;
 // cout<<"cov=";
   for (unsigned int i=0;i<6;i++)
   	 for (unsigned int j=0;j<6;j++){
   	 	 Q(i,j)=msg->pose.covariance[k];
        // ROS_INFO("%f ",msg->pose.covariance[6*i+j]);
         k++;
   	 }
   	//cout<<"LQ= "<<Q<<endl;
  mcc.process(lidar_d,Q,1);
  ekf.process(lidar_d,Q,0);
  VectorXd prediction = mcc.get();
  VectorXd prediction_na = ekf.get();


  geometry_msgs::PoseStamped mcc_data;
  mcc_data.pose.position.x=prediction(0);
  mcc_data.pose.position.y=prediction(1);
  mcc_data.pose.position.z=2+prediction(2);
  mcc_path.header.frame_id="paths";
  mcc_path.header.stamp=ros::Time::now();
  mcc_path.poses.push_back(mcc_data);
  mcc_slam_pub.publish(mcc_path);
  //cout<<"from lidar="<<prediction<<endl;

  nav_msgs::Odometry mcc_odom;
  mcc_odom.header.stamp = ros::Time::now();
  mcc_odom.header.frame_id = "paths";
  mcc_odom.pose.pose.position.x=1+prediction(0);
  mcc_odom.pose.pose.position.y=1+prediction(1);
  mcc_odom.pose.pose.position.z=1+prediction(2);
  mcc_odom.twist.twist.linear.x = prediction(3);
  mcc_odom.twist.twist.linear.y = prediction(4);
  mcc_odom.twist.twist.angular.z = prediction(5);
  mcc_odom_pub.publish(mcc_odom);


   nav_msgs::Odometry lidar_odom;
  lidar_odom.header.stamp = ros::Time::now();
  lidar_odom.header.frame_id = "paths";
  lidar_odom.pose.pose.position.x=data.pose.position.x;
  lidar_odom.pose.pose.position.y=data.pose.position.y;
  lidar_odom.pose.pose.position.z=data.pose.position.z;
  //mcc_odom.twist.twist.linear.x = prediction(3);
  //mcc_odom.twist.twist.linear.y = prediction(4);
  //mcc_odom.twist.twist.angular.z = prediction(5)
  lidar_odom_pub.publish(lidar_odom);



  lidar_path.header.frame_id="paths";
  lidar_path.header.stamp=ros::Time::now();
  geometry_msgs::PoseStamped lidar_data;
  lidar_data.pose.position.x=d_vals(0);
  lidar_data.pose.position.y=d_vals(1);
  lidar_data.pose.position.z=d_vals(2);
  lidar_path.poses.push_back(lidar_data);
  lidar_path_pub.publish(lidar_path);

 nav_msgs::Odometry mccnoattack_odom;
  mccnoattack_odom.header.stamp = ros::Time::now();
  mccnoattack_odom.header.frame_id = "paths";
  mccnoattack_odom.pose.pose.position.x=data.pose.position.x;
  mccnoattack_odom.pose.pose.position.y=data.pose.position.y;
  mccnoattack_odom.pose.pose.position.z=data.pose.position.z;
  noattack_pub.publish(mccnoattack_odom);

nav_msgs::Odometry ekf_odom;
  ekf_odom.header.stamp = ros::Time::now();
  ekf_odom.header.frame_id = "paths";
  ekf_odom.pose.pose.position.x=prediction_na(0);
  ekf_odom.pose.pose.position.y=prediction_na(1);
  ekf_odom.pose.pose.position.z=prediction_na(2);
  ekf_odom_pub.publish(ekf_odom);
 

geometry_msgs::PoseStamped mcc_noatt_data;
  mcc_noatt_data.pose.position.x=prediction_na(0);
  mcc_noatt_data.pose.position.y=prediction_na(1)-5;
  mcc_noatt_data.pose.position.z=prediction_na(2);
  ekf_path.header.frame_id="paths";
  ekf_path.header.stamp=ros::Time::now();
  ekf_path.poses.push_back(mcc_noatt_data);
  ekf_slam_pub.publish(ekf_path);
  //data.pose.twist.linear.x=prediction(3);
  
}
void StereoOdomToPath(const nav_msgs::Odometry::ConstPtr& msg)
{
  // ROS_INFO("Seq: [%d]", msg->header.seq);
  // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
  // ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  // ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);
  

  geometry_msgs::PoseStamped data;
  data.pose.position.x=msg->pose.pose.position.x;
  data.pose.position.y=msg->pose.pose.position.y;
  data.pose.position.z=msg->pose.pose.position.z;
  //double deg=90.0;
  //tf2::Quaternion q_orig,q_rot,q_new;
  //tf2::convert(data.pose.orientation,q_orig);
  //double r=3.14159, p=0, y=0;
   //q_rot.setRPY(r, p, y);
   //q_new = q_rot*q_orig;
    //q_new.normalize();
  // tf2::convert(q_new,data.pose.orientation);
  data.pose.orientation.x=msg->pose.pose.orientation.x;
  data.pose.orientation.y=msg->pose.pose.orientation.y;
  data.pose.orientation.z=msg->pose.pose.orientation.z;
  data.pose.orientation.w=msg->pose.pose.orientation.w;
  
  long int timestamp=duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  
  VectorXd d_vals(3);
  
  d_vals<<data.pose.position.x,data.pose.position.y,data.pose.position.z;

  //cout<<d_vals<<endl;
  stereo_d.set(timestamp,DataPointType::STEREO,d_vals);
  // cout<<"cov=";
   MatrixXd Q(6,6);
   int k=0;
   for (unsigned int i=0;i<6;i++)
   	 for (unsigned int j=0;j<6;j++){
   	 	 Q(i,j)=msg->pose.covariance[k];
         //ROS_INFO("%f ",msg->pose.covariance[6*i+j]);
         k++;
   	 }
   	//cout<<"Q= "<<Q<<endl;
  

  stereo_path.header.frame_id="paths";
  stereo_path.header.stamp=ros::Time::now();
  //std::cout<<ros::Time::now()<<endl;
  stereo_path.poses.push_back(data);
  stereo_path_pub.publish(stereo_path);

 // ekf.process(stereo_d,Q);
 // noatt.process(stereo_d,Q);
  //VectorXd prediction = ekf.get();
  //VectorXd prediction_na = noatt.get();
 // // cout<<"from stereo="<<prediction<<endl;

 //  geometry_msgs::PoseStamped mcc_data;
 //  mcc_data.pose.position.x=prediction(0);
 //  mcc_data.pose.position.y=1+prediction(1);
 //  mcc_data.pose.position.z=prediction(2);
 //  mcc_path.header.frame_id="paths";
 //  mcc_path.header.stamp=ros::Time::now();
 //  mcc_path.poses.push_back(mcc_data);
 //  mcc_slam_pub.publish(mcc_path);

 //  nav_msgs::Odometry mcc_odom;
 //  mcc_odom.header.stamp = ros::Time::now();
 //  mcc_odom.header.frame_id = "paths";
 //  mcc_odom.pose.pose.position.x=1+prediction(0);
 //  mcc_odom.pose.pose.position.y=1+prediction(1);
 //  mcc_odom.pose.pose.position.z=prediction(2);
 //  mcc_odom.twist.twist.linear.x = prediction(3);
 //  mcc_odom.twist.twist.linear.y = prediction(4);
 //  mcc_odom.twist.twist.angular.z = prediction(5);
 //  mcc_odom_pub.publish(mcc_odom);

  

}
uint64_t timeSinceEpochMillisec() {
  
  return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}


void GTToPath(const nav_msgs::Odometry::ConstPtr& msg)
{
  // ROS_INFO("Seq: [%d]", msg->header.seq);
  // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
  // ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  // ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);
  
  geometry_msgs::PoseStamped data;
  data.pose.position.x=msg->pose.pose.position.x;
  data.pose.position.y=msg->pose.pose.position.y;
  data.pose.position.z=msg->pose.pose.position.z;
  //double deg=90.0;
  //tf2::Quaternion q_orig,q_rot,q_new;
  //tf2::convert(data.pose.orientation,q_orig);
  //double r=3.14159, p=0, y=0;
   //q_rot.setRPY(r, p, y);
   //q_new = q_rot*q_orig;
    //q_new.normalize();
  // tf2::convert(q_new,data.pose.orientation);
  data.pose.orientation.x=msg->pose.pose.orientation.x;
  data.pose.orientation.y=msg->pose.pose.orientation.y;
  data.pose.orientation.z=msg->pose.pose.orientation.z;
  data.pose.orientation.w=msg->pose.pose.orientation.w;
  
  //long long int timestamp=timeSinceEpochMillisec();
  //std::cout << timestamp << std::endl;

  gt_path.header.frame_id="paths";
  gt_path.header.stamp=ros::Time::now();
  gt_path.poses.push_back(data);
  gt_path_pub.publish(gt_path);

}



int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "ekf_slam");

  
  ros::NodeHandle n;
  
  
  ros::Subscriber ls= n.subscribe("rtabmap_lidar/odom", 1000, LidarOdomToPath);
  ros::Subscriber ss= n.subscribe("stereo_odom", 1000, StereoOdomToPath);
  ros::Subscriber gts= n.subscribe("base_pose_ground_truth", 1000, GTToPath);
  
  mcc_odom_pub = n.advertise<nav_msgs::Odometry>("/mcc_odom", 1000);
  lidar_odom_pub = n.advertise<nav_msgs::Odometry>("/lidar_odom", 1000);
  noattack_pub = n.advertise<nav_msgs::Odometry>("/noattack_odom", 1000);

  ekf_odom_pub = n.advertise<nav_msgs::Odometry>("/ekf_odom", 1000);
  
  lidar_path_pub = n.advertise<nav_msgs::Path>("/lidar_path", 1000);
  stereo_path_pub=n.advertise<nav_msgs::Path>("/stereo_path", 1000);
  gt_path_pub=n.advertise<nav_msgs::Path>("/gt_path", 1000);
  mcc_slam_pub=n.advertise<nav_msgs::Path>("/mcc_path", 1000);
  ekf_slam_pub=n.advertise<nav_msgs::Path>("/ekf_path", 1000);

  while (ros::ok())
  {
  	 //char c=getchar();
     //if(c=='a') cout<<"found";
  	 ros::spin();
  }
 

  return 0;
}
