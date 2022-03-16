#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src, gradX, gradY;

double harris(Mat& Gx, Mat& Gy, int x, int y, int w, Mat& H)
{
  // A vous de l'écrire
  return 0;  // doit en fait retourner det(H)-0.15 tr^2(H)
}


void mouse_callback(int event, int x, int y, int flags, void* unused)
{
if (event == EVENT_LBUTTONDOWN)
  {
	cout << "Vous avez cliqué sur le point (" << x << ", " << y << ")\n";

	// A vous de compléter	
  }
}





int main( int argc, char** argv )
{
  Mat tmp = imread(argv[1]);
  Mat kx;

  if (tmp.channels() > 1)
    {
      cvtColor( tmp, src, COLOR_BGR2GRAY );
      tmp= src;
    }
  // tmp.convertTo(src, CV_8UC1 );
  tmp.convertTo(src, CV_8UC1);
  // attention, autre conversion en CV_32FC1 probablement nécessaire...


  // Routine de traitement du clic souris
  setMouseCallback("src", mouse_callback, NULL);

  // Détecteur de Harris-Laplace
  Ptr<xfeatures2d::HarrisLaplaceFeatureDetector> p= xfeatures2d::HarrisLaplaceFeatureDetector::create();
  vector<KeyPoint> kpts;
  
  p->detect(src, kpts);

  Mat res;
  drawKeypoints(src, kpts, res);

  namedWindow("res");
  imshow("res", res);

  waitKey();
  return 0;
}


