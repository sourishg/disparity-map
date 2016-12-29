#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>

using namespace cv;
using namespace std;

Mat Q, XR, XT;

void publishPointCloud(Mat& left, Mat& dmap, ofstream& obj_file) {
  Mat V = Mat(4, 1, CV_64FC1);
  Mat pos = Mat(4, 1, CV_64FC1);
  int ndisp = 40;
  for (int i = 0; i < dmap.cols; i++) {
    for (int j = 0; j < dmap.rows; j++) {
      int d = dmap.at<uchar>(j,i) * ((float)ndisp / 255.);
      // if low disparity, then ignore
      if (d < 2)
        continue;
      // V is the vector to be multiplied to Q to get
      // the 3D homogenous coordinates of the image point
      V.at<double>(0,0) = (double)(i);
      V.at<double>(1,0) = (double)(j);
      V.at<double>(2,0) = (double)d;
      V.at<double>(3,0) = 1.;
      pos = Q * V; // 3D homogeneous coordinate
      double X = pos.at<double>(0,0) / pos.at<double>(3,0);
      double Y = pos.at<double>(1,0) / pos.at<double>(3,0);
      double Z = pos.at<double>(2,0) / pos.at<double>(3,0);
      Mat point3d_cam = Mat(3, 1, CV_64FC1);
      point3d_cam.at<double>(0,0) = X;
      point3d_cam.at<double>(1,0) = Y;
      point3d_cam.at<double>(2,0) = Z;
      // transform 3D point from camera frame to robot frame
      Mat point3d_robot = XR * point3d_cam + XT;
      int r = (int)left.at<Vec3b>(j,i)[2];
      int g = (int)left.at<Vec3b>(j,i)[1];
      int b = (int)left.at<Vec3b>(j,i)[0];
      obj_file << "v " << point3d_robot.at<double>(0,0) << " " <<
      point3d_robot.at<double>(1,0) << " " << point3d_robot.at<double>(2,0);
      obj_file << " " << r << " " << g << " " << b << endl;
    }
  }
}

int main(int argc, char const *argv[])
{
  ofstream obj_file;
  Mat img_left = imread(argv[1], 1);
  Mat img_disp = imread(argv[2], 0);
  cv::FileStorage fs1(argv[3], cv::FileStorage::READ);
  fs1["Q"] >> Q; // depth to disparity mapping matrix
  fs1["XR"] >> XR; // rotation from camera frame to robot frame
  fs1["XT"] >> XT; // translation from camera frame to robot frame
  obj_file.open(argv[4]);
  publishPointCloud(img_left, img_disp, obj_file);
  return 0;
}