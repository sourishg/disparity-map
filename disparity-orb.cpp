#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_disp;
Mat img_left_desc, img_right_desc;
vector< KeyPoint > kpl, kpr;
int w = 0;

bool inImg(Mat& img, int x, int y) {
  if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
    return true;
}

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

long descCost(Point leftpt, Point rightpt, int w) {
  int x0r = kpr[0].pt.x;
  int y0r = kpr[0].pt.y;
  int ynr = kpr[kpr.size()-1].pt.y;
  int x0l = kpl[0].pt.x;
  int y0l = kpl[0].pt.y;
  int ynl = kpl[kpl.size()-1].pt.y;
  long cost = 0;
  for (int j = -w; j <= w; j++) {
    for (int k = -w; k <= w; k++) {
      if (!isLeftKeyPoint(leftpt.x+j, leftpt.y+k) || 
          !isRightKeyPoint(rightpt.x+j, rightpt.y+k))
        continue;
      int idxl = (leftpt.x+j-x0l)*(ynl-y0l+1)+(leftpt.y+k-y0l);
      int idxr = (rightpt.x+j-x0r)*(ynr-y0r+1)+(rightpt.y+k-y0r);
      cost += norm(img_left_desc.row(idxl), img_right_desc.row(idxr), CV_L1);
    }
  }
  return cost / ((2*w+1)*(2*w+1));
}

double descCostNCC(Point leftpt, Point rightpt, int w) {
  int x0r = kpr[0].pt.x;
  int y0r = kpr[0].pt.y;
  int ynr = kpr[kpr.size()-1].pt.y;
  int x0l = kpl[0].pt.x;
  int y0l = kpl[0].pt.y;
  int ynl = kpl[kpl.size()-1].pt.y;
  double costL = 0;
  double costR = 0;
  double cost = 0;
  int idxl0 = (leftpt.x-x0l)*(ynl-y0l+1)+(leftpt.y-y0l);
  int idxr0 = (rightpt.x-x0r)*(ynr-y0r+1)+(rightpt.y-y0r);
  for (int j = -w; j <= w; j++) {
    for (int k = -w; k <= w; k++) {
      if (!isLeftKeyPoint(leftpt.x+j, leftpt.y+k) || 
          !isRightKeyPoint(rightpt.x+j, rightpt.y+k))
        continue;
      int idxl = (leftpt.x+j-x0l)*(ynl-y0l+1)+(leftpt.y+k-y0l);
      int idxr = (rightpt.x+j-x0r)*(ynr-y0r+1)+(rightpt.y+k-y0r);
      double d1 = norm(img_left_desc.row(idxl), img_left_desc.row(idxl0), 
                       CV_L1);
      double d2 = norm(img_right_desc.row(idxr), img_right_desc.row(idxr0), 
                       CV_L1);
      costL += d1*d1;
      costR += d2*d2;
      cost += d1*d2;
    }
  }
  cost /= (sqrt(costL) * sqrt(costR));
  cout << "ncc: " << cost << endl;
  return cost;
}

int getCorresPointRight(Point p, int ndisp) {
  long minCost = 1e9;
  int chosen_i = 0;
  for (int i = p.x-ndisp; i <= p.x; i++) {
    long cost = descCost(p, Point(i,p.y), w);
    if (cost < minCost) {
      minCost = cost;
      chosen_i = i;
    }
  }
  if (minCost == 0)
    return p.x;
  return chosen_i;
  /*
  double corr = -10;
  int chosen_i = 0;
  for (int i = p.x-ndisp; i <= p.x; i++) {
    double cost = descCostNCC(p, Point(i,p.y), w);
    if (cost > corr) {
      corr = cost;
      chosen_i = i;
    }
  }
  cout << "corr: " << corr << endl;
  return chosen_i;
  */
}

int getCorresPointLeft(Point p, int ndisp) {
  long minCost = 1e9;
  int chosen_i = 0;
  for (int i = p.x; i <= p.x+ndisp; i++) {
    long cost = descCost(Point(i,p.y), p, w);
    if (cost < minCost) {
      minCost = cost;
      chosen_i = i;
    }
  }
  if (minCost == 0)
    return p.x;
  return chosen_i;
}

void computeDisparityMapORB(int ndisp) {
  img_disp = Mat(img_left.rows, img_left.cols, CV_8UC1, Scalar(0));
  for (int i = ndisp+1; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      cout << i << ", " << j << endl;
      if (!isLeftKeyPoint(i,j))
        continue;
      int right_i = getCorresPointRight(Point(i,j), ndisp);
      // left-right check
      /*
      int left_i = getCorresPointLeft(Point(right_i,j), ndisp);
      if (abs(left_i-i) > 4)
        continue;
      */
      int disparity = abs(i - right_i);
      img_disp.at<uchar>(j,i) = disparity;
    }
  }
}

void cacheDescriptorVals() {
  OrbDescriptorExtractor extractor;
  //BriefDescriptorExtractor extractor;
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      kpl.push_back(KeyPoint(i,j,1));
      kpr.push_back(KeyPoint(i,j,1));
    }
  }
  extractor.compute(img_left, kpl, img_left_desc);
  extractor.compute(img_right, kpr, img_right_desc);
}

void preprocess(Mat& img) {
  Mat dst;
  bilateralFilter(img, dst, 10, 15, 15);
  img = dst.clone();
}

int main(int argc, char const *argv[])
{
  img_left = imread(argv[1], 1);
  img_right = imread(argv[2], 1);
  //preprocess(img_left);
  //preprocess(img_right);
  cacheDescriptorVals();
  computeDisparityMapORB(40);
  //namedWindow("IMG-LEFT", 1);
  //namedWindow("IMG-RIGHT", 1);
  while (1) {
    imshow("IMG-LEFT", img_left);
    imshow("IMG-RIGHT", img_right);
    imshow("IMG-DISP", img_disp);
    if (waitKey(30) > 0) {
      imwrite(argv[3], img_disp);
      break;
    }
  }
  return 0;
}