#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// 사각형 꼭짓점 TopLeft, TopRight, BottomRight, BottomLeft 순서로 정렬
vector<Point2f> orderPoints(const vector<Point2f>& points) {
	vector<Point2f> sortedPoints(4);

	// x + y 합계로 TL, BR 찾기
	// TL (TopLeft) : x+y가 가장 작음
	// BR (BottomRight) : x+y가 가장 큼
	vector<float> sums;
	for (const auto& p : points) {
		sums.push_back(p.x + p.y);
	}
	auto minMaxSum = minmax_element(sums.begin(), sums.end());
	sortedPoints[0] = points[distance(sums.begin(), minMaxSum.first)]; // TL
	sortedPoints[2] = points[distance(sums.begin(), minMaxSum.second)]; // BR

	// x - y 차이로 TR, BL 찾기
	// TR (TopRight) : x-y가 가장 큼
	// BL (BottomLeft) : x-y가 가장 작음
	vector<float> diffs;
	for (const auto& p : points) {
		diffs.push_back(p.x - p.y);
	}
	auto minMaxDiff = minmax_element(diffs.begin(), diffs.end());
	sortedPoints[1] = points[distance(diffs.begin(), minMaxDiff.second)]; // TR
	sortedPoints[3] = points[distance(diffs.begin(), minMaxDiff.first)]; // BL

	return sortedPoints;
}

// 당구대로 생각되는 사각형 찾기
vector<Point2f> findDocumentContour(const Mat& image) {
	Mat gray, blurred, edged;

	// 1. 그레이스케일(어차피 contour인식하는데는 이미지 색은 상관없기 때문에) 및 노이즈 제거(생각보다 백그라운드에 노이즈가 꽤 있어서)
	cvtColor(image, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blurred, Size(5, 5), 0);

	// 2. Canny 엣지 검출
	Canny(blurred, edged, 75, 200);

	// dilation으로 끊어진 엣지들 연결
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(edged, edged, kernel);

	// 3. 윤곽선 찾기
	vector<vector<Point>> contours;
	findContours(edged, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// 가장 큰 윤곽선을 찾기 위한 변수
	double maxArea = 0;
	vector<Point> bestContour;
	vector<Point2f> bestQuad;

	// 4. 모든 윤곽선을 순회하며 사각형 필터링
	for (const auto& contour : contours) {
		double area = contourArea(contour);

		// 너무 작은 윤곽선은 무시(이 파라미터도 조정해봐야할듯, 리사이즈한 이미지에 대해서 적절한 값으로 한거라)
		if (area < 2000) continue;

		// 5. 다각형으로 근사 (Contour를 단순화)
		double peri = arcLength(contour, true);
		vector<Point> approxPoly;
		approxPolyDP(contour, approxPoly, 0.02 * peri, true);

		// 6. 4개의 꼭짓점을 가지고, 면적이 가장 큰 것을 선택
		if (approxPoly.size() == 4 && area > maxArea) {
			maxArea = area;
			bestContour = approxPoly;
		}
	}

	// `std::vector<cv::Point>`를 `std::vector<cv::Point2f>`로 변환
	if (!bestContour.empty()) {
		for (const auto& p : bestContour) {
			bestQuad.push_back(Point2f(p.x, p.y));
		}
	}

	return bestQuad;
}

int main() {
	// 1. 이미지 로드 (컬러)
	Mat image_color = imread("Test1_TableAndBallsOnly.png", IMREAD_COLOR);
	if (image_color.empty()) {
		cerr << "이미지없음" << endl;
		return -1;
	}

	// display용 복사본
	Mat image_display = image_color.clone();

	// 2. 당구대로 생각되는 사각형 찾기
	vector<Point2f> srcPoints = findDocumentContour(image_color);

	if (srcPoints.empty()) {
		cerr << "maxArea 파라미터 조정 필요" << endl;
		return -1;
	}

	cout << "contour 찾기 서어공" << endl;

	// 3. 찾은 꼭짓점 정렬 (TL, TR, BR, BL)
	srcPoints = orderPoints(srcPoints);

	// 4. 잘 보려고 리사이징 후 꼭짓점 대입
	float outputWidth = 500;
	float outputHeight = outputWidth * 1.414;

	vector<Point2f> dstPoints = {
		Point2f(0, 0),
		Point2f(outputWidth - 1, 0),
		Point2f(outputWidth - 1, outputHeight - 1),
		Point2f(0, outputHeight - 1)
	};

	// 5. 호모그래피 계산
	Mat H = getPerspectiveTransform(srcPoints, dstPoints);

	// 6. warping
	Mat warpedImage;
	warpPerspective(image_color, warpedImage, H, Size(outputWidth, outputHeight));

	// 7. 결과 시각화 (찾은 윤곽선 그리기)
	vector<vector<Point>> contourToDraw;
	vector<Point> intPoints;
	for (const auto& p : srcPoints) {
		intPoints.push_back(Point(p.x, p.y));
	}
	contourToDraw.push_back(intPoints);
	drawContours(image_display, contourToDraw, 0, Scalar(0, 255, 0), 1);

	// 8. 결과 창 표시 (원본 크기)
	imshow("Original with Contour", image_display);
	imshow("Top Down View", warpedImage);

	imwrite("contour_output.jpg", image_display);
	imwrite("topdown_output.jpg", warpedImage);

	waitKey(0);

	return 0;
}