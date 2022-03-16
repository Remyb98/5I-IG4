#include <fstream>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <string.h>
#include <time.h>

using namespace cv;
using namespace std;

void epilignes(InputArray u, InputArray &F, Mat &img, int i)
{
    vector<Point3f> lines;
    computeCorrespondEpilines(u, i, F, lines);
    int cols = img.cols;
    for (int i = 0; i < lines.size(); ++i) {
        Point3f p = lines[i];
        line(img, Point(0, -p.z / p.y), Point(cols, (-p.z - p.x * cols) / p.y), Scalar::all(-1));
    }
}

double epierror(Mat F, vector<Point2f> U, vector<Point2f> V)
{
    double epsilon = 0;

    // à compléter
    return epsilon / U.size();
}

int main(int argc, char *argv[])
{
    Mat image[2];
    Size boardSize, imageSize;
    vector<Point2f> points[2];
    bool found;

    // load + display image
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " imageL imageR ...\n";
        return 1;
    }

    // Le programme peut en réaliter traiter plusieurs paires d'images, qui
    // doivent toujours être dans l'ordre: gauche1, droite1, gauche2, droite2,...
    for (int i = 1; i < argc; i++) {
        int j = (i - 1) % 2;
        vector<Point2f> pointbuf;

        image[j] = imread(argv[i], IMREAD_COLOR);
        imageSize = image[j].size();

        boardSize.width = 8;
        boardSize.height = 5;

        // find the chessboard
        found = findChessboardCorners(image[j], boardSize, pointbuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // improve the found corners' coordinate accuracy
            Mat viewGray;

            cvtColor(image[j], viewGray, COLOR_BGR2GRAY);
            cornerSubPix(viewGray, pointbuf, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

            // ajouter les points trouvés
            points[j].insert(points[j].end(), pointbuf.begin(), pointbuf.end());
        } else {
            cerr << "Chessboard not found.\n";
            return 1;
        }
    }

    Mat F = findFundamentalMat(points[0], points[1]);
    epilignes(points[0], F, image[1], 1);
    epilignes(points[1], F, image[0], 2);
    
    namedWindow("gauche", WINDOW_AUTOSIZE);
    imshow("gauche", image[0]);

    namedWindow("droite", WINDOW_AUTOSIZE);
    imshow("droite", image[1]);

    waitKey(0);
    return 0;
}
