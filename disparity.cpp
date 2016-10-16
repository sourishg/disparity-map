#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_disp;

bool inImg(Mat& img, int x, int y) {
  if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
    return true;
}

long costF(Mat& left, Mat& right) {
  long cost = 0;
  for (int i = 0; i < 32; i++) {
    cost += abs(left.at<uchar>(0,i)-right.at<uchar>(0,i));
  }
  return cost;
}

int getCorresPointORB(Point p, Mat& desc, Mat& img, int maxd) {
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

int getCorresPoint(Point p, Mat& img1, Mat& img2, int maxd) {
  int w = 6;
  long minCost = 1e9;
  int chosen_i = 0;
  for (int i = max(0,p.x-maxd); i < min(img2.cols,p.x+maxd); i++) {
    long error = 0;
    for (int k = -w; k <= w; k++) {
      for (int j = -w; j <= w; j++) {
        if (inImg(img1, p.x+k, p.y+j) && inImg(img2, i+k, p.y+j)) {
          for (int ch = 0; ch < 3; ch++) {
            error += abs(img1.at<Vec3b>(p.y+j,p.x+k)[ch] - 
                         img2.at<Vec3b>(p.y+j,i+k)[ch]);
          }
        }
      }
    }
    if (error < minCost) {
      minCost = error;
      chosen_i = i;
    }
  }
  return chosen_i;
}

void computeDisparityMapORB() {
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
      int right_i = getCorresPointORB(Point(i,j), desc_left, img_right, 40);
      if (right_i == 0)
        continue;
      /*
      // left-right check
      Mat desc_right;
      vector<KeyPoint> kpr;
      kpr.push_back(KeyPoint(right_i,j,1));
      extractor.compute(img_right, kpr, desc_right);
      int left_i = getCorresPointORB(Point(right_i,j), desc_right, img_left, 
                                    40);
      if (abs(left_i-i) > 5)
        continue;
      */
      int disparity = abs(i - right_i);
      img_disp.at<uchar>(j,i) = disparity * (255. / 40.);
    }
  }
}

void computeDisparityMap() {
  img_disp = Mat(img_left.rows, img_left.cols, CV_8UC1, Scalar(0));
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      cout << i << ", " << j << endl;
      int valid_pixel = 0;
      for (int ch = 0; ch < 3; ch++) {
        valid_pixel += img_left.at<Vec3b>(j,i)[ch];
      }
      if (valid_pixel == 0)
        continue;
      int right_i = getCorresPoint(Point(i,j), img_left, img_right, 40);
      int left_i = getCorresPoint(Point(right_i,j),img_right,img_left,40);
      if (abs(left_i-i) > 5)
        continue;
      int disparity = abs(i - right_i);
      img_disp.at<uchar>(j,i) = disparity * (255. / 40.);
    }
  }
}

int main(int argc, char const *argv[])
{
  img_left = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  img_right = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  computeDisparityMap();
  namedWindow("IMG-LEFT", 1);
  namedWindow("IMG-RIGHT", 1);
  while (1) {
    imshow("IMG-LEFT", img_left);
    imshow("IMG-RIGHT", img_right);
    imshow("IMG-DISP", img_disp);
    if (waitKey(30) > 0) {
      imwrite("disp.png", img_disp);
      break;
    }
  }
  return 0;
}