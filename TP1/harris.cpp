#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src, gradX, gradY;

void showMatrix(Mat& M)
{
  for (int i = 0; i < M.rows; ++i) {
    for (int j = 0; j < M.cols; ++j) {
      cout << M.at<double>(i, j) << " ";
    }
    cout << endl;
  }
}

double harris(Mat& Gx, Mat& Gy, int x, int y, int w, Mat& H)
{
  H = Mat(2, 2, CV_32FC1);
  for (int i = 0; i < H.rows; ++i) {
    for (int j = 0; j < H.cols; ++j) {
      H.at<double>(i, j) = 0;
    }
  }

  for (int j = y - w; j < y + w; ++j) {
    for (int i = x - w; i < x + w; ++i) {
      Mat u = Mat(2, 1, CV_32FC1);
      u.at<double>(0) = Gx.at<double>(j, i);
      u.at<double>(1) = Gy.at<double>(j, i);

      H += u * u.t();
    }
  }
  // A vous de l'écrire
  // doit en fait retourner det(H)-0.15 tr^2(H)
  return determinant(H) - .15 * trace(H)[0];
}


void mouse_callback(int event, int x, int y, int flags, void* unused)
{
if (event == EVENT_LBUTTONDOWN)
  {
	cout << endl << endl << "Vous avez cliqué sur le point (" << x << ", " << y << ")\n";
  Mat H;
  Mat values;
  Mat vectors;
  double score = harris(gradX, gradY, x, y, 3, H);
  eigen(H, values, vectors);
  cout << "Score Harris: " << score << endl << endl;
  cout << "Matrix Harris:" << endl;
  showMatrix(H);
  cout << endl << "Vecteurs propres:" << endl;
  showMatrix(vectors);
  cout << endl << "Valeurs propres:" << endl;
  showMatrix(values);
  }
}


void getDoGX(Mat& K, int w, double sigma)
{
  K= Mat(2*w+1,2*w+1, CV_64FC1);

  double alpha= 1/(2*M_PI*pow(sigma,4));
  double beta= -1/(2*sigma*sigma);
  
  for (int i=-w; i <= w; i++)
    for (int j=-w; j<= w; j++)
      K.at<double>(j+w,i+w)= i*alpha*exp(beta*(i*i+j*j));
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main( int argc, char** argv )
{
  Mat tmp = imread(argv[1]);
  Mat kx;

  if (tmp.channels() > 1)
    {
      cvtColor( tmp, src, COLOR_BGR2GRAY );
      tmp = src;
    }
  tmp.convertTo(tmp, CV_8UC1);
  tmp.convertTo(tmp, CV_32FC1);

  getDoGX(kx, 3, 2);

  filter2D(tmp, gradX, -1, kx);
  filter2D(tmp, gradY, -1, kx.t());
  normalize(gradX, gradX, 1, 0, NORM_MINMAX);
  normalize(gradY, gradY, 1, 0, NORM_MINMAX);

  filter2D(tmp, tmp, -1, kx);
  normalize(tmp, tmp, 1, 0, NORM_MINMAX);

  namedWindow("src");
  imshow("src", tmp);

  // Routine de traitement du clic souris
  setMouseCallback("src", mouse_callback, NULL);

  // A vous de compléter le code...
  
  waitKey();
  return 0;
}
