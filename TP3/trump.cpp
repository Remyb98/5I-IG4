#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

void mouse_callback(int event, int x, int y, int flags, void *unused)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        cout << "Vous avez cliqué sur le point (" << x << ", " << y << ")\n";
    }
}

int main(int, char **argv)
{
    // Charger les images
    Mat panneau = imread("panneau.jpg", 1);
    Mat trump = imread("trump.jpg", 1);

    vector<Point2f> psrc, pdst;

    // Trump dimension => 732 x 1008
    psrc.push_back(Point2f(0, 0));
    psrc.push_back(Point2f(732, 0));
    psrc.push_back(Point2f(732, 1008));
    psrc.push_back(Point2f(0, 1008));

    // Panneau
    pdst.push_back(Point2f(500, 100));
    pdst.push_back(Point2f(726, 77));
    pdst.push_back(Point2f(762, 464));
    pdst.push_back(Point2f(471, 488));

    Mat H = findHomography(psrc, pdst);

    cout << H << endl;

    Mat out;

    warpPerspective(trump, out, H, Size(panneau.cols, panneau.rows));

    for (int i = 0; i < out.rows; ++i) {
        for (int j = 0; j < out.cols; ++j) {
            Vec3b pixel = out.at<Vec3b>(i, j);
            if (0 == pixel[0] + pixel[1] + pixel[2]) {
                out.at<Vec3b>(i, j) = panneau.at<Vec3b>(i, j);
            }
        }
    }

    namedWindow("resultat", WINDOW_AUTOSIZE);
    imshow("resultat", out);

    imwrite("resultat.jpg", out);
    setMouseCallback("resultat", mouse_callback, NULL);

    // Attendre une touche
    waitKey(0);

    return 0;
}
