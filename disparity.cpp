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

int getCorresPoint(Point p, Mat& img1, Mat& img2, int ndisp) {
  int w = 7;
  long minCost = 1e9;
  int chosen_i = 0;
  for (int i = p.x - ndisp; i <= p.x; i++) {
    long error = 0;
    for (int k = -w; k <= w; k++) {
      for (int j = -w; j <= w; j++) {
        if (inImg(img1, p.x+k, p.y+j) && inImg(img2, i+k, p.y+j)) {
          error += abs(img1.at<uchar>(p.y+j,p.x+k) - 
                       img2.at<uchar>(p.y+j,i+k));
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

void computeDisparityMap(int ndisp) {
  img_disp = Mat(img_left.rows, img_left.cols, CV_8UC1, Scalar(0));
  for (int i = ndisp; i < img_left.cols-ndisp; i++) {
    for (int j = ndisp; j < img_left.rows-ndisp; j++) {
      cout << i << ", " << j << endl;
      /*
      int valid_pixel = 0;
      for (int ch = 0; ch < 3; ch++) {
        valid_pixel += img_left.at<Vec3b>(j,i)[ch];
      }
      */
      /*
      if (img_left.at<uchar>(j,i) == 0)
        continue;
      */
      int right_i = getCorresPoint(Point(i,j), img_left, img_right, ndisp);
      int disparity = abs(i - right_i);
      img_disp.at<uchar>(j,i) = disparity * (255. / ndisp);
    }
  }
}

int main(int argc, char const *argv[])
{
  img_left = imread(argv[1], 0);
  img_right = imread(argv[2], 0);
  //computeDisparityMapORB(20);
  computeDisparityMap(20);
  namedWindow("IMG-LEFT", 1);
  namedWindow("IMG-RIGHT", 1);
  while (1) {
    imshow("IMG-LEFT", img_left);
    imshow("IMG-RIGHT", img_right);
    imshow("IMG-DISP", img_disp);
    if (waitKey(30) > 0) {
      //imwrite(argv[3], img_disp);
      break;
    }
  }
  return 0;
}