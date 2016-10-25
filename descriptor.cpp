#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_left_disp, img_right_disp;
Mat img_left_desc, img_right_desc;
vector< KeyPoint > kpl, kpr;

bool isLeftKeyPoint(int i, int j) {
  int n = kpl.size();
  return (i >= kpl[0].pt.x && i <= kpl[n-1].pt.x
          && j >= kpl[0].pt.y && j <= kpl[n-1].pt.y);
}

bool isRightKeyPoint(int i, int j) {
  int n = kpr.size();
  return (i >= kpr[0].pt.x && i <= kpr[n-1].pt.x
          && j >= kpr[0].pt.y && j <= kpr[n-1].pt.y);
}

long costF(const Mat& left, const Mat& right) {
  long cost = 0;
  for (int i = 0; i < 32; i++) {
    cost += abs(left.at<uchar>(0,i)-right.at<uchar>(0,i));
  }
  return cost;
}

int getCorresPoint(Point p, Mat& img, int ndisp) {
  int w = 2;
  long minCost = 1e9;
  int chosen_i = 0;
  int x0r = kpr[0].pt.x;
  int y0r = kpr[0].pt.y;
  int ynr = kpr[kpr.size()-1].pt.y;
  int x0l = kpl[0].pt.x;
  int y0l = kpl[0].pt.y;
  int ynl = kpl[kpl.size()-1].pt.y;
  for (int i = p.x-ndisp; i <= p.x; i++) {
    long cost = -1;
    for (int j = -w; j <= w; j++) {
      for (int k = -w; k <= w; k++) {
        if (!isLeftKeyPoint(p.x+j, p.y+k) || !isRightKeyPoint(i+j, p.y+k))
          continue;
        int idxl = (p.x+j-x0l)*(ynl-y0l+1)+(p.y+k-y0l);
        int idxr = (i+j-x0r)*(ynr-y0r+1)+(p.y+k-y0r);
        cost += costF(img_left_desc.row(idxl), img_right_desc.row(idxr));
      }
    }
    if (cost < minCost) {
      minCost = cost;
      chosen_i = i;
    }
  }
  cout << "minCost: " << minCost << endl;
  return chosen_i;
}

void mouseClickRight(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
  }
}

void mouseClickLeft(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    if (!isLeftKeyPoint(x,y))
      return;
    int right_i = getCorresPoint(Point(x,y), img_right, 20);
    /*
    Mat desc_right;
    vector<KeyPoint> kpr;
    kpr.push_back(KeyPoint(right_i,y,1));
    extractor.compute(img_right, kpr, desc_right);
    int left_i = getCorresPoint(Point(right_i,y), desc_right, img_left, 70);
    cout << "Left right diff: " << abs(left_i-x) << endl;
    if (abs(left_i-x) > 5)
      return;
    */
    circle(img_left_disp, Point(x,y), 3, Scalar(255,0,0), 1, 8, 0);
    circle(img_right_disp, Point(right_i,y), 3, Scalar(255,0,0), 1, 8, 0);
    cout << "Left: " << x << " Right: " << right_i << endl;
  }
}

void cacheDescriptorVals() {
  OrbDescriptorExtractor extractor;
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      kpl.push_back(KeyPoint(i,j,1));
      kpr.push_back(KeyPoint(i,j,1));
    }
  }
  extractor.compute(img_left, kpl, img_left_desc);
  extractor.compute(img_right, kpr, img_right_desc);
}

int main(int argc, char const *argv[])
{
  img_left = imread(argv[1]);
  img_right = imread(argv[2]);
  img_left_disp = imread(argv[1]);
  img_right_disp = imread(argv[2]);
  //OrbFeatureDetector detector(500, 1.2f, 8, 15, ORB::HARRIS_SCORE);
  //detector.detect(img, kp);
  cacheDescriptorVals();
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