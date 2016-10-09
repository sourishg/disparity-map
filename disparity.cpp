#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_disp;

long costF(Mat& left, Mat& right) {
  long cost = 0;
  for (int i = 0; i < 32; i++) {
    cost += abs(left.at<uchar>(0,i)-right.at<uchar>(0,i));
  }
  return cost;
}

int getCorresPoint(Point p, Mat& desc, Mat& img, int maxd) {
  long minCost = 1e9;
  int chosen_i = 0;
  OrbDescriptorExtractor extractor;
  Mat desc2;
  for (int i = max(0,p.x-maxd); i < min(img.cols,p.x+maxd); i++) {
    vector<KeyPoint> kp;
    kp.push_back(KeyPoint(i,p.y,1));
    extractor.compute(img, kp, desc2);
    if (desc2.empty())
      continue;
    long cost = costF(desc, desc2);
    if (cost < minCost) {
      minCost = cost;
      chosen_i = i;
    }
  }
  return chosen_i;
}

void computeDisparityMap() {
  img_disp = Mat(img_left.rows, img_left.cols, CV_8UC1, Scalar(0));
  OrbDescriptorExtractor extractor;
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      cout << i << ", " << j << endl;
      vector<KeyPoint> kp;
      kp.push_back(KeyPoint(i,j,1));
      Mat desc_left;
      extractor.compute(img_left, kp, desc_left);
      if (desc_left.empty())
        continue;
      int right_i = getCorresPoint(Point(i,j), desc_left, img_right, 40);
      if (right_i == 0)
        continue;
      /*
      // left-right check
      Mat desc_right;
      vector<KeyPoint> kpr;
      kpr.push_back(KeyPoint(right_i,j,1));
      extractor.compute(img_right, kpr, desc_right);
      int left_i = getCorresPoint(Point(right_i,j), desc_right, img_left, 40);
      if (abs(left_i-i) > 5)
        continue;
      */
      int disparity = abs(i - right_i);
      img_disp.at<uchar>(j,i) = disparity * (255. / 40.);
    }
  }
}

int main(int argc, char const *argv[])
{
  img_left = imread(argv[1]);
  img_right = imread(argv[2]);
  computeDisparityMap();
  namedWindow("IMG-LEFT", 1);
  namedWindow("IMG-RIGHT", 1);
  while (1) {
    imshow("IMG-LEFT", img_left);
    imshow("IMG-RIGHT", img_right);
    imshow("IMG-DISP", img_disp);
    if (waitKey(30) > 0) {
      break;
    }
  }
  return 0;
}