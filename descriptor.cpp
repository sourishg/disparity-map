#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_left_disp, img_right_disp;
int cur_left_x, cur_left_y;
Mat desc_left, desc_right;

long costF(Mat& left, Mat& right) {
  long cost = 0;
  for (int i = 0; i < 32; i++) {
    cost += abs(left.at<uchar>(0,i)-right.at<uchar>(0,i));
  }
  return cost;
}

void mouseClickRight(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    cout << "------- CALLED COST --------" << endl;
    OrbDescriptorExtractor extractor;
    vector<KeyPoint> kp;
    Mat desc_right;
    kp.push_back(KeyPoint(x, y, 1));
    extractor.compute(img_right, kp, desc_right);
    cout << "Left: " << cur_left_x << ", " << cur_left_y << endl;
    cout << "Right: " << x << ", " << y << endl;
    cout << "Left desc: " << desc_left << endl;
    cout << "Right desc: " << desc_right << endl;
    cout << "Cost: " << costF(desc_left, desc_right) << endl;
  }
}

void mouseClickLeft(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    /*
    cout << "------- CHANGED REF --------" << endl;
    cur_left_x = x;
    cur_left_y = y;
    OrbDescriptorExtractor extractor;
    vector<KeyPoint> kp;
    kp.push_back(KeyPoint(x, y, 1));
    extractor.compute(img_left, kp, desc_left);
    cout << "Changed left ref: " << x << ", " << y << endl;
    */
    OrbDescriptorExtractor extractor;
    vector<KeyPoint> kp;
    kp.push_back(KeyPoint(x, y, 1));
    extractor.compute(img_left, kp, desc_left);
    if (desc_left.empty())
      return;
    circle(img_left_disp, Point(x,y), 3, Scalar(255,0,0), 1, 8, 0);
    long minCost = 1e9;
    int chosen_i = 0;
    for (int i = 0; i < img_left.cols; i++) {
      vector<KeyPoint> kp_r;
      kp_r.push_back(KeyPoint(i,y,1));
      extractor.compute(img_right, kp_r, desc_right);
      if (desc_right.empty())
        continue;
      long cost = costF(desc_left, desc_right);
      if (cost < minCost) {
        minCost = cost;
        chosen_i = i;
      }
    }
    if (chosen_i != 0) {
      circle(img_right_disp, Point(chosen_i,y), 3, Scalar(255,0,0), 1, 8, 0);
    }
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