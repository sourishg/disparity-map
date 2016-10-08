#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

Mat img;

void mouseClick(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    cout << "Called" << endl;
    OrbDescriptorExtractor extractor;
    vector<KeyPoint> kp;
    Mat desc;
    kp.push_back(KeyPoint(x, y, 1));
    extractor.compute(img, kp, desc);
    cout << x << ", " << y << endl;
    cout << desc << endl;
  }
}

int main(int argc, char const *argv[])
{
  img = imread(argv[1]);
  //OrbFeatureDetector detector(500, 1.2f, 8, 15, ORB::HARRIS_SCORE);
  //detector.detect(img, kp);
  namedWindow("IMG", 1);
  setMouseCallback("IMG", mouseClick, NULL);
  while (1) {
    imshow("IMG", img);
    if (waitKey(30) > 0) {
      break;
    }
  }
  return 0;
}