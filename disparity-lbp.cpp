#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_disp;
int niter = 30;
int window_size = 5;
int ndisp = 20;
int smooth_lambda = 20;
int smooth_trunc = 5;

const int MAX_LABELS = 70;

enum DIR {
  LEFT,
  RIGHT,
  UP,
  DOWN,
  COST
};

struct Pixel {
  int best_label;
  double msg[5][MAX_LABELS];
};

struct MarkovRandomField {
  int width, height;
  vector< Pixel > img;
  MarkovRandomField() {
    width = img_left.cols;
    height = img_left.rows;
    int tsize = width * height;
    img.resize(tsize);
    for (int i = 0; i < tsize; i++) {
      for (int label = 0; label < ndisp; label++) {
        for (int k = 0; k < 5; k++) {
          img[i].best_label = 0;
          img[i].msg[k][label] = 1;
        }
      }
    }
  }
};

long costFunc(Point p, int d) {
  long cost = 0;
  for (int i = -window_size; i <= window_size; i++) {
    for (int j = -window_size; j <= window_size; j++) {
      cost += abs(img_left.at<uchar>(j+p.y,i+p.x) - 
                  img_right.at<uchar>(j+p.y,i+p.x-d));
    }
  }
  return cost/((2*window_size+1)*(2*window_size+1));
}

long smoothFunc(int i, int j) {
  return (long)smooth_lambda * min(abs(i-j), smooth_trunc);
}

void initCost(MarkovRandomField& mrf) {
  int border = ndisp;
  for (int i = border; i < mrf.width-border; i++) {
    for (int j = border; j < mrf.height-border; j++) {
      for (int label = 0; label < ndisp; label++) {
        mrf.img[j*img_left.cols+i].msg[COST][label] = costFunc(Point(i,j), 
        label);
      }
    }
  }
}

void passMessage(MarkovRandomField& mrf, int x, int y, DIR dir) {
  double updated_msg[MAX_LABELS];
  double norm_const = 0;
  int width = mrf.width;
  for (int i = 0; i < ndisp; i++) {
    double max_val = -1;
    for (int j = 0; j < ndisp; j++) {
      double cost = exp(-smoothFunc(i,j));
      cost *= exp(-mrf.img[y*width+x].msg[COST][j]);
      if (dir != LEFT)
        cost *= mrf.img[y*width+x].msg[LEFT][j];
      if (dir != RIGHT)
        cost *= mrf.img[y*width+x].msg[RIGHT][j];
      if (dir != UP)
        cost *= mrf.img[y*width+x].msg[UP][j];
      if (dir != DOWN)
        cost *= mrf.img[y*width+x].msg[DOWN][j];
      max_val = max(max_val, cost);
    }
    updated_msg[i] = max_val;
    norm_const += max_val;
  }
  for (int i = 0; i < ndisp; i++) {
    switch (dir) {
      case LEFT:
        mrf.img[y*width+x-1].msg[RIGHT][i] = updated_msg[i] / norm_const;
        break;
      case RIGHT:
        mrf.img[y*width+x+1].msg[LEFT][i] = updated_msg[i] / norm_const;
        break;
      case UP:
        mrf.img[(y-1)*width+x].msg[DOWN][i] = updated_msg[i] / norm_const;
        break;
      case DOWN:
        mrf.img[(y+1)*width+x].msg[UP][i] = updated_msg[i] / norm_const;
        break;
      default:
        assert(0);
        break;
    }
  }
}

void propagate(MarkovRandomField& mrf, DIR dir) {
  int width = mrf.width;
  int height = mrf.height;
  
  switch (dir) {
    case LEFT:
      for (int i = width-1; i > 0; i--) {
        for (int j = 0; j < height; j++) {
          passMessage(mrf, i, j, dir);
        }
      }
      break;
    case RIGHT:
      for (int i = 0; i < width-1; i++) {
        for (int j = 0; j < height; j++) {
          passMessage(mrf, i, j, dir);
        }
      }
      break;
    case DOWN:
      for (int i = 0; i < width; i++) {
        for (int j = 0; j < height-1; j++) {
          passMessage(mrf, i, j, dir);
        }
      }
      break;
    case UP:
      for (int i = 0; i < width-1; i++) {
        for (int j = height-1; j > 0; j--) {
          passMessage(mrf, i, j, dir);
        }
      }
      break;
    default:
      assert(0);
      break;
  }
}

double MAP(MarkovRandomField& mrf) {
  int width = mrf.width;
  int height = mrf.height;
  for (int i = 0; i < mrf.img.size(); i++) {
    double max_belief = -1;
    for (int k = 0; k < ndisp; k++) {
      double belief = 1;
      belief *= exp(-mrf.img[i].msg[COST][k]);
      belief *= mrf.img[i].msg[LEFT][k];
      belief *= mrf.img[i].msg[RIGHT][k];
      belief *= mrf.img[i].msg[UP][k];
      belief *= mrf.img[i].msg[DOWN][k];
      if (belief > max_belief) {
        max_belief = belief;
        mrf.img[i].best_label = k;
      }
    }
  }
  double energy = 0;
  
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      int d = mrf.img[j*width+i].best_label;
      energy += mrf.img[j*width+i].msg[COST][d];
      if (i-1 >= 0)
        energy += smoothFunc(d, mrf.img[j*width+i-1].best_label);
      if (i+1 < width)
        energy += smoothFunc(d, mrf.img[j*width+i+1].best_label);
      if (j-1 >= 0)
        energy += smoothFunc(d, mrf.img[(j-1)*width+i].best_label);
      if (j+1 < height)
        energy += smoothFunc(d, mrf.img[(j+1)*width+i].best_label);
    }
  }
  
  return energy;
}

void computeDisparityMap(MarkovRandomField& mrf) {
  initCost(mrf);
  for (int i = 0; i < niter; i++) {
    propagate(mrf, RIGHT);
    propagate(mrf, DOWN);
    propagate(mrf, LEFT);
    propagate(mrf, UP);
    
    double energy = MAP(mrf);
    cout << "Iter " << (i+1) << ": " << energy << endl;
  }
  int border = ndisp;
  for (int i = border; i < mrf.width-border; i++) {
    for (int j = border; j < mrf.height-border; j++) {
      img_disp.at<uchar>(j,i) = mrf.img[j*mrf.width+i].best_label * (256 / 
ndisp);
    }
  }
}

int main(int argc, char const *argv[])
{
  img_left = imread(argv[1], 0);
  img_right = imread(argv[2], 0);
  img_disp = Mat(img_left.rows, img_left.cols, CV_8UC1, Scalar(0));
  namedWindow("IMG-LEFT", 1);
  namedWindow("IMG-RIGHT", 1);
  
  MarkovRandomField mrf;
  computeDisparityMap(mrf);
  
  imwrite(argv[3], img_disp);
  return 0;
  
  while (1) {
    imshow("IMG-LEFT", img_left);
    imshow("IMG-RIGHT", img_right);
    imshow("IMG-DISP", img_disp);
    if (waitKey(30) > 0) {
      //imwrite("disp-lbp.png", img_disp);
      break;
    }
  }
  return 0;
}