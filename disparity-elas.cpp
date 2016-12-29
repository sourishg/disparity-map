#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "elas.h"
#include "image.h"
#include "popt_pp.h"

using namespace std;
using namespace cv;

Mat generateDisparityMap(Mat& left, Mat& right) {
  Mat lb, rb;
  if (left.empty() || right.empty()) 
    return left;

  // convert images to grayscale
  cvtColor(left, lb, CV_BGR2GRAY);
  cvtColor(right, rb, CV_BGR2GRAY);

  const Size imsize = lb.size();
  const int32_t dims[3] = {imsize.width,imsize.height,imsize.width};

  Mat leftdpf = Mat::zeros(imsize, CV_32F);
  Mat rightdpf = Mat::zeros(imsize, CV_32F);

  Elas::parameters param;
  param.postprocess_only_left = true;
  Elas elas(param);
  
  elas.process(lb.data,rb.data,leftdpf.ptr<float>(0),rightdpf.ptr<
float>(0),dims);

  // normalize disparity values between 0 and 255
  int max_disp = -1;
  for (int i = 0; i < imsize.width; i++) {
    for (int j = 0; j < imsize.height; j++) {
      if (leftdpf.at<uchar>(j,i) > max_disp) max_disp = leftdpf.at<uchar>(j,i);
    }
  }
  for (int i = 0; i < imsize.width; i++) {
    for (int j = 0; j < imsize.height; j++) {
      leftdpf.at<uchar>(j,i) = 
(int)max(255.0*(float)leftdpf.at<uchar>(j,i)/max_disp,0.0);
    }
  }
  
  Mat show = Mat(left.rows, left.cols, CV_8UC1, Scalar(0));
  leftdpf.convertTo(show, CV_8U, 5.);
  /*
  max_disp = -1;
  for (int i = 0; i < imsize.width; i++) {
    for (int j = 0; j < imsize.height; j++) {
      if (show.at<uchar>(j,i) > max_disp) max_disp = show.at<uchar>(j,i);
    }
  }
  for (int i = 0; i < imsize.width; i++) {
    for (int j = 0; j < imsize.height; j++) {
      show.at<uchar>(j,i) = 
(int)max(255.0*(float)show.at<uchar>(j,i)/max_disp,0.0);
    }
  }
  */
  return show;
}

int main(int argc, char **argv) {
  Mat img_left = imread(argv[1], 1);
  Mat img_right = imread(argv[2], 1);
  Mat img_disp = generateDisparityMap(img_left, img_right);
  imwrite(argv[3], img_disp);
  return 0;
  while (1) {
    imshow("IMG-LEFT", img_left);
    imshow("IMG-RIGHT", img_right);
    if (!img_disp.empty())
      imshow("IMG-DISP", img_disp);
    if (waitKey(30) > 0) {
      //imwrite(argv[3], img_disp);
      break;
    }
  }
  return 0;
}