#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

static void onMouse(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    vector<Point>* pPoints = static_cast<vector<Point>*>(userdata);

    if (pPoints->size() < 4) {
      pPoints->emplace_back(x, y);
      cout << pPoints->size() << "번째 Point(" << x << ", " << y << ")" << endl;
    }
  }
}

vector<Point> getFourPoints(const Mat& inputImage) {
  vector<Point> points;
  Mat tempImage;
  const string windowName =
      "네 꼭짓점 클릭(세로로 긴 테이블 기준, 왼쪽 위부터 CW 방향, Q로 종료)";

  namedWindow(windowName);
  setMouseCallback(windowName, onMouse, &points);

  while (points.size() < 4) {
    tempImage = inputImage.clone();

    for (int i = 0; i < points.size(); i++) {
      circle(tempImage, points[i], 5, Scalar(0, 255, 0), -1);
      putText(tempImage, to_string(i + 1), points[i] + Point(10, 0),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
    }

    imshow(windowName, tempImage);

    int key = waitKey(30);
    if (key == 'q' || key == 'Q') {
      cout << "사용자가 입력을 취소했습니다." << endl;
      break;
    }
  }

  destroyWindow(windowName);
  waitKey(1);
  return points;
}

void changeHomography(const Mat& src, Mat& dst, const vector<Point>& points,
                      const Size& outputSize) {
  vector<Point2f> srcPoints;
  for (const Point& p : points) {
    srcPoints.push_back(static_cast<Point2f>(p));
  }

  vector<Point2f> dstPoints;
  dstPoints.emplace_back(0, 0);
  dstPoints.emplace_back(outputSize.width - 1, 0);
  dstPoints.emplace_back(outputSize.width - 1, outputSize.height - 1);
  dstPoints.emplace_back(0, outputSize.height - 1);

  Mat perspectiveMatrix = getPerspectiveTransform(srcPoints, dstPoints);
  warpPerspective(src, dst, perspectiveMatrix, outputSize);
}

int main() {
  const float scaleFactor = 0.3f;
  const float boardWidth = 1'442.5f;
  const float boardHeight = 2'845.0f;
  const float ballDiameter = 61.5f;

  Mat image = imread("../resource/20251008_181035 (중형).jpg");
  if (image.empty()) {
    cout << "이미지 못 불러옴" << endl;
    return 1;
  }

  // 4개 포인트 입력 받기
  // vector<Point> selectedPoints = getFourPoints(image);
  vector<Point> selectedPoints = {
      {265, 506}, {497, 511}, {716, 947}, {10, 937}};

  if (selectedPoints.size() != 4) {
    cout << "모든 포인트 입력을 받지 않고 종료." << endl;
    return 1;
  }

  // 호모그래피 변환
  const float outputWidth = boardWidth * scaleFactor;
  const float outputHeight = boardHeight * scaleFactor;
  Mat transformedImage;
  changeHomography(image, transformedImage, selectedPoints,
                   Size(outputWidth, outputHeight));

  // k-means (k=4)
  Mat image_32f;
  transformedImage.convertTo(image_32f, CV_32F);

  Mat samples(transformedImage.rows * transformedImage.cols, 3, CV_32F);
  samples = image_32f.reshape(1, transformedImage.rows * transformedImage.cols);

  const int K = 4;
  Mat labels, centers;

  TermCriteria criteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 100, 1.0);

  kmeans(samples, K, labels, criteria, 500, KMEANS_PP_CENTERS, centers);

  Mat resultImage(transformedImage.size(), transformedImage.type());
  for (int y = 0; y < transformedImage.rows; y++)
    for (int x = 0; x < transformedImage.cols; x++) {
      int clusterIndex = labels.at<int>(y * transformedImage.cols + x);

      float b = centers.at<float>(clusterIndex, 0);
      float g = centers.at<float>(clusterIndex, 1);
      float rValue = centers.at<float>(clusterIndex, 2);

      resultImage.at<Vec3b>(y, x) =
          Vec3b(static_cast<uchar>(b), static_cast<uchar>(g),
                static_cast<uchar>(rValue));
    }

  // imshow("Homography Transformed", transformedImage);
  imshow("K-Means Result", resultImage);
  char key = waitKey(1);
  while (!(key == 'q' || key == 'Q'))
    key = waitKey(1);  // Q키 입력 들어올 때까지 무한 대기
  return 0;
}