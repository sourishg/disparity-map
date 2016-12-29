#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "daisy/daisy.h"

using namespace cv;
using namespace std;
using namespace kutility;

int im_width, im_height;
Mat leftDesc, rightDesc;
int w = 2;

bool inImg(int x, int y) {
  if (x >= 0 && x < im_width && y >= 0 && y < im_height)
    return true;
}

double descCost(Point leftpt, Point rightpt, int w) {
  double cost = 0;
  for (int j = -w; j <= w; j++) {
    for (int k = -w; k <= w; k++) {
      if (!inImg(leftpt.x+j, leftpt.y+k) || 
          !inImg(rightpt.x+j, rightpt.y+k))
        continue;
      int idxl = (leftpt.x+j)+(leftpt.y+k)*im_width;
      int idxr = (rightpt.x+j)+(rightpt.y+k)*im_width;
      cost += norm(leftDesc.row(idxl), rightDesc.row(idxr), CV_L1);
    }
  }
  return cost;
}

int getCorresPointRight(Point p, int ndisp) {
  double minCost = 1e9;
  int chosen_i = 0;
  for (int i = p.x-ndisp; i <= p.x; i++) {
    double cost = descCost(p, Point(i,p.y), w);
    if (cost < minCost) {
      minCost = cost;
      chosen_i = i;
    }
  }
  if (fabs(minCost) < 0.00001)
    return p.x;
  return chosen_i;
}

int getCorresPointLeft(Point p, int ndisp) {
  double minCost = 1e9;
  int chosen_i = 0;
  for (int i = p.x; i <= p.x+ndisp; i++) {
    double cost = descCost(Point(i,p.y), p, w);
    if (cost < minCost) {
      minCost = cost;
      chosen_i = i;
    }
  }
  if (fabs(minCost) < 0.00001)
    return p.x;
  return chosen_i;
}

void computeDisparityMap(int ndisp, Mat &img_disp) {
  img_disp = Mat(im_height, im_width, CV_8UC1, Scalar(0));
  for (int i = ndisp+1; i < im_width; i++) {
    for (int j = 0; j < im_height; j++) {
      cout << i << ", " << j << endl;
      int right_i = getCorresPointRight(Point(i,j), ndisp);
      // left-right check
      int left_i = getCorresPointLeft(Point(right_i,j), ndisp);
      if (abs(left_i-i) > 4)
        continue;
      int disparity = abs(i - right_i);
      img_disp.at<uchar>(j,i) = disparity;
    }
  }
}

void computeDenseDesc(const Mat &img, Mat &descrpOut) {
  // daisy params
  int verbose_level = 4;
  int rad   = 20;
  int radq  =  5;
  int thq   =  8;
  int histq =  8;
  int nrm_type = NRM_FULL;
  bool disable_interpolation = false;

  // associate pointer
  uchar *im = img.data;
  int h = img.rows;
  int w = img.cols;

  daisy* desc = new daisy();
  desc->set_image(im, h, w);
  desc->verbose(verbose_level);
  desc->set_parameters(rad, radq, thq, histq);
  desc->set_normalization(NRM_FULL);
  desc->initialize_single_descriptor_mode();
  desc->compute_descriptors();
  desc->normalize_descriptors();
  
  int iy, ix, descSize;
  descSize = desc->descriptor_size();
  descrpOut.create(h*w, descSize, CV_32FC1);
  for (iy=0; iy<h; ++iy)
  {
    for (ix=0; ix<w; ++ix)
    {
      float* thor = NULL;
      desc->get_descriptor(iy, ix, thor);
      memcpy(descrpOut.ptr(iy*w+ix), thor, descSize*sizeof(float));
    }
  }
}

void preprocess(Mat& img) {
  Mat dst;
  bilateralFilter(img, dst, 10, 15, 15);
  img = dst.clone();
}

void normalizeDisparity(Mat &disp) {
  int max_disp = -1;
  for (int i = 0; i < disp.cols; i++) {
    for (int j = 0; j < disp.rows; j++) {
      if ((int)disp.at<uchar>(j,i) > max_disp)
        max_disp = disp.at<uchar>(j,i);
    }
  }
  for (int i = 0; i < disp.cols; i++) {
    for (int j = 0; j < disp.rows; j++) {
      disp.at<uchar>(j,i) *= (255. / (float)max_disp);
    }
  }
}

int main(int argc, char** argv) {
  /*
  Mat imgL = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  Mat imgR = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  //preprocess(imgL);
  //preprocess(imgR);
  im_width = imgL.cols;
  im_height = imgL.rows;
  computeDenseDesc(imgL, leftDesc);
  computeDenseDesc(imgR, rightDesc);
  Mat disp;
  computeDisparityMap(40, disp);
  imshow("left", imgL);
  imshow("disp", disp);
  imwrite(argv[3], disp);
  waitKey(0);
  */
  Mat disp = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  normalizeDisparity(disp);
  imwrite(argv[1], disp);
  return 0;
}