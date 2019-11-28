#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "usagecheck.h"
#include "datapoint.h"
#include "tools.h"
#include "fusionekf.h"

using namespace std;
using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char* argv[]) {

  /*******************************************************************
   * CHECK IF CORRECTLY EXECUTED BY USER
   *************************************
   ******************************/
 
  check_arguments(argc, argv);
for (int sig_itr=1;sig_itr<6;sig_itr++)
 {
  string in_filename = argv[1];
  string out_filename = argv[2];
  string att_fname;
  att_fname="attack_out_" + std::to_string(sig_itr)+".txt";
  out_filename="out_ekf_" + std::to_string(sig_itr)+".txt";
  ifstream in_file(in_filename.c_str(), ifstream::in);
  ofstream out_file(out_filename.c_str(), ofstream::out);
  ofstream attack_out(att_fname,ofstream::out);
 // ofstream sig_itr_f("rmse_sigma.txt",ofstream::app);
  check_files(in_file, in_filename, out_file, out_filename);

  /*******************************************************************
   * READ DATA FROM FILE AND STORE IN MEMORY
   *******************************************************************/
  vector<DataPoint> all_sensor_data;
  vector<DataPoint> all_truth_data;

  double val1, val2, val3;
  double x, y, vx, vy;
  long long timestamp;
  string sensor_id;

  string line;
  int itr=0,litr=0;
  //////////////////////////
  vector<int> my_vec;
  int a=rand()%(sig_itr*5)+1;
  for (int i=0;i<a;i++)
  {
    int b=rand()%100+1;
    my_vec.push_back(b);
    cout<<my_vec[i]<<endl;
  }
  ///////////////////////////
  while(getline(in_file, line)){

    istringstream iss(line);
    ostringstream att;
    DataPoint sensor_data;
    DataPoint truth_data;
    itr++;
   // std::cout<<"itr= "<<itr<<"\n";
    iss >> sensor_id;
    attack_out<<sensor_id<<"\t";  
    if (sensor_id.compare("L") == 0){

      iss >> val1;
      iss >> val2;
      iss >> timestamp;
       litr++;
      VectorXd lidar_vec(2);
      for (int j=0;j<my_vec.size();j++)
      {
        if(litr==my_vec[j])
           {
            //val1+=0.0;
            //val2+=10.0;
           }
      }
      //if( (litr>50 && litr<=60)|| (litr>300&&litr<330))
      //{ 
         //std::cout<<"itr= "<<itr<<"\n";
        //val1+=10.0;val2+=10.0;
      //}
      lidar_vec << val1, val2;
      attack_out<<val1<<"\t";
      attack_out<<val2<<"\t";
      attack_out<<timestamp<<"\t";  
      sensor_data.set(timestamp, DataPointType::LIDAR, lidar_vec);

    } else if (sensor_id.compare("R") == 0){

      iss >> val1;
      iss >> val2;
      iss >> val3;
      iss >> timestamp;


      VectorXd radar_vec(3);
      radar_vec << val1, val2, val3;
      sensor_data.set(timestamp, DataPointType::RADAR, radar_vec);
      attack_out<<val1<<"\t";
      attack_out<<val2<<"\t";
      attack_out<<val3<<"\t";
      attack_out<<timestamp<<"\t";  
    }

        
    iss >> x;
    iss >> y;
    iss >> vx;
    iss >> vy;
   
   attack_out<<x<<"\t";
      attack_out<<y<<"\t";
      attack_out<<vx<<"\t";
      attack_out<<vy<<"\n";

    VectorXd truth_vec(4);
    truth_vec << x, y, vx, vy;
    truth_data.set(timestamp, DataPointType::STATE, truth_vec);

    all_sensor_data.push_back(sensor_data);
    all_truth_data.push_back(truth_data);
  }
  std::cout<<"Lidar dcount= "<<litr;

  /*******************************************************************
   * USE DATA AND FUSIONEKF FOR STATE ESTIMATIONS
   *******************************************************************/
   FusionEKF fusionEKF;

   vector<VectorXd> estimations;
   vector<VectorXd> ground_truths;

  for (int k = 0; k < all_sensor_data.size(); ++k)
  {

    //fusionEKF.sigma_val=sig_itr;
    fusionEKF.process(all_sensor_data[k]);

    VectorXd prediction = fusionEKF.get();
    VectorXd measurement = all_sensor_data[k].get_state();
    VectorXd truth =  all_truth_data[k].get();

    out_file << prediction(0) << "\t";
    out_file << prediction(1) << "\t";
    out_file << prediction(2) << "\t";
    out_file << prediction(3) << "\t";

    out_file << measurement(0) << "\t";
    out_file << measurement(1) << "\t";

    out_file << truth(0) << "\t";
    out_file << truth(1) << "\t";
    out_file << truth(2) << "\t";
    out_file << truth(3) << "\n";

    estimations.push_back(prediction);
    ground_truths.push_back(truth);
  }

  /*******************************************************************
   * CALCULATE ROOT MEAN SQUARE ERROR
   *******************************************************************/
   VectorXd RMSE = calculate_RMSE(estimations, ground_truths);
   cout << "Accuracy - RMSE:" << endl;
   cout << RMSE << endl;
   //sig_itr_f<<sig_itr<<"\t"<<RMSE(0)<<"\t"<<RMSE(1)<<"\t"<<RMSE(2)<<"\t"<<RMSE(3)<<"\n";

  /*******************************************************************
   * PRINT TO CONSOLE IN A NICE FORMAT FOR DEBUGGING
   *******************************************************************/
   //print_EKF_data(RMSE, estimations, ground_truths, all_sensor_data);

  /*******************************************************************
   * CLOSE FILES
   *******************************************************************/
  if (out_file.is_open()) { out_file.close(); }
  if (in_file.is_open()) { in_file.close(); } 
 }

  return 0;
}
