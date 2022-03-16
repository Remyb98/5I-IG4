#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
  Mat a= Mat(2,1,CV_32FC1);
  //  Mat b= Mat(2,2,CV_64FC1);

  //  a.at<>(0,0)= 1;
  //  a.at<double>(0,1)= 2;
  //  a.at<double>(1,0)= 3;
  //  a.at<double>(1,1)= 4;

  cout << "a= " << a << "\n";

   try
     {
       cout << a*a.t();
     }
   catch (Exception e)
     {
       cout << "erreur " << e.code << ", err=" << e.err << ", msg=" << e.msg << "\n";
     }
  //  b=b*a;

  return 0;
}
