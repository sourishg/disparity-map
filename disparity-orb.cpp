#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_disp;
Mat img_left_desc, img_right_desc;
vector< KeyPoint > kpl, kpr;

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

long costF(const Mat& left, const Mat& right) {
  long cost = 0;
  for (int i = 0; i < 32; i++) {
    cost += abs(left.at<uchar>(0,i)-right.at<uchar>(0,i));
  }
  return cost;
}

int getCorresPointRight(Point p, int ndisp) {
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
    long cost = 0;
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
  if (minCost == 0)
    return p.x;
  return chosen_i;
}

int getCorresPointLeft(Point p, int ndisp) {
  int w = 2;
  long minCost = 1e9;
  int chosen_i = 0;
  int x0r = kpr[0].pt.x;
  int y0r = kpr[0].pt.y;
  int ynr = kpr[kpr.size()-1].pt.y;
  int x0l = kpl[0].pt.x;
  int y0l = kpl[0].pt.y;
  int ynl = kpl[kpl.size()-1].pt.y;
  for (int i = p.x; i <= p.x+ndisp; i++) {
    long cost = 0;
    for (int j = -w; j <= w; j++) {
      for (int k = -w; k <= w; k++) {
        if (!isRightKeyPoint(p.x+j, p.y+k) || !isLeftKeyPoint(i+j, p.y+k))
          continue;
        int idxr = (p.x+j-x0l)*(ynl-y0l+1)+(p.y+k-y0l);
        int idxl = (i+j-x0r)*(ynr-y0r+1)+(p.y+k-y0r);
        cost += costF(img_left_desc.row(idxl), img_right_desc.row(idxr));
      }
    }
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
  int x0 = kpl[0].pt.x;
  int y0 = kpl[0].pt.y;
  int yn = kpl[kpl.size()-1].pt.y;
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      cout << i << ", " << j << endl;
      if (!isLeftKeyPoint(i,j))
        continue;
      int right_i = getCorresPointRight(Point(i,j), ndisp);
      // left-right check
      int left_i = getCorresPointLeft(Point(right_i,j), ndisp);
      if (abs(left_i-i) > 5)
        continue;
      int disparity = abs(i - right_i);
      img_disp.at<uchar>(j,i) = disparity * (255. / ndisp);
    }
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
  img_left = imread(argv[1], 1);
  img_right = imread(argv[2], 1);
  cacheDescriptorVals();
  computeDisparityMapORB(20);
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