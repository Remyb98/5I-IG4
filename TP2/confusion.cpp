#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <iomanip>

#define N 30

using namespace cv;
using namespace std;

Mat src[2];


double moyenne(vector<string> images)
{
  vector<KeyPoint> kpts[2];
  Mat out[2];
  
  for (int i = 0; i <= 1; i++)
    {
        src[i] = imread(images[i]);
        cvtColor(src[i], src[i], COLOR_BGR2GRAY);
        Ptr<Feature2D> sift = SIFT::create();
        sift->detectAndCompute(src[i], Mat(), kpts[i], out[i]);
    }

    BFMatcher matcher(NORM_L2, true);
    vector<DMatch> matches;

    matcher.match(out[0], out[1], matches, Mat());

    std::sort(matches.begin(), matches.end());

    double moy = 0;

    for (int i = 0; i < N; ++i) {
        moy += matches[i].distance;
    }

    return moy /= N;
}

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        for (int j = 1; j < argc; ++j) {
            cout << setprecision(6) << setw(18) << moyenne(vector<string> { argv[i], argv[j] });
        }
        cout << endl;
    }

    return 0;
}
