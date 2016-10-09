#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_left_disp, img_right_disp;

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

void mouseClickRight(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
  }
}

void mouseClickLeft(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    OrbDescriptorExtractor extractor;
    vector<KeyPoint> kp;
    Mat desc_left;
    kp.push_back(KeyPoint(x, y, 1));
    extractor.compute(img_left, kp, desc_left);
    if (desc_left.empty())
      return;
    int right_i = getCorresPoint(Point(x,y), desc_left, img_right, 70);
    if (right_i == 0)
      return;
    Mat desc_right;
    vector<KeyPoint> kpr;
    kpr.push_back(KeyPoint(right_i,y,1));
    extractor.compute(img_right, kpr, desc_right);
    int left_i = getCorresPoint(Point(right_i,y), desc_right, img_left, 70);
    cout << "Left right diff: " << abs(left_i-x) << endl;
    if (abs(left_i-x) > 5)
      return;
    circle(img_left_disp, Point(x,y), 3, Scalar(255,0,0), 1, 8, 0);
    circle(img_right_disp, Point(right_i,y), 3, Scalar(255,0,0), 1, 8, 0);
  }
}

int main(int argc, char const *argv[])
{
  img_left = imread(argv[1]);
  img_right = imread(argv[2]);
  img_left_disp = imread(argv[1]);
  img_right_disp = imread(argv[2]);
  //OrbFeatureDetector detector(500, 1.2f, 8, 15, ORB::HARRIS_SCORE);
  //detector.detect(img, kp);
  namedWindow("IMG-LEFT", 1);
  namedWindow("IMG-RIGHT", 1);
  setMouseCallback("IMG-LEFT", mouseClickLeft, NULL);
  setMouseCallback("IMG-RIGHT", mouseClickRight, NULL);
  while (1) {
    imshow("IMG-LEFT", img_left_disp);
    imshow("IMG-RIGHT", img_right_disp);
    if (waitKey(30) > 0) {
      break;
    }
  }
  return 0;
}