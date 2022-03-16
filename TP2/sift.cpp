#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <iomanip>

#define N 10

using namespace cv;
using namespace std;

Mat src[2];


int main(int argc, char** argv)
{
  vector<KeyPoint> kpts[2];
  Mat out[2];
  
  for (int i = 0; i <= 1; i++)
    {
      src[i] = imread(argv[1 + i], IMREAD_GRAYSCALE);
      resize(src[i], src[i], Size(), .25, .25);
      Ptr<Feature2D> sift = SIFT::create();
      sift->detectAndCompute(src[i], Mat(), kpts[i], out[i]);
      drawKeypoints(src[i], kpts[i], src[i], Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }

    BFMatcher matcher(NORM_L2, true);
    vector<DMatch> matches;
    vector<char> matchesMask;

    matcher.match(out[0], out[1], matches, Mat());

    std::sort(matches.begin(), matches.end());

    float min, max, moy;

    moy = 0;
    for (int i = 0; i < matches.size(); ++i) {
      matchesMask.push_back((i < N)? 1 : 0);
      if (i < N) {
        moy += matches[i].distance;
        max = matches[i].distance;
      }
    }

    min = matches.front().distance;
    moy /= N;

    cout << setprecision(6) << "min: " << min << endl << "max: " << max << endl << "moy: " << moy << endl;

    Mat res;
    drawMatches(src[0], kpts[0], src[1], kpts[1], matches, res, Scalar::all(-1), Scalar::all(-1), matchesMask, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imshow("res", res);

    waitKey();
    return 0;
}
