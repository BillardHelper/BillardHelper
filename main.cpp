#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(void){
    Mat img = imread("test_img1.jpg");

    imshow("test", img);

    waitKey(0);

    return 0;
}