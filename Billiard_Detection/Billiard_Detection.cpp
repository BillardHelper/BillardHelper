#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;

cv::Mat img, hsvImage;
vector<cv::Point> tableCorners;

// Step 1
bool loadAndConvertImage(const string &filename)
{
    img = cv::imread(filename);
    if (img.empty())
        return false;
    cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);
    return true;
}

// Step 2
void maskTableArea()
{
    cv::Scalar lowerGreen(115, 180, 160), upperGreen(123, 255, 255);
    // cv::Scalar lowerBlue(110, 100, 100), upperBlue(130, 255, 255);
    cv::Mat mask;
    cv::inRange(hsvImage, lowerGreen, upperGreen, mask);
    cv::erode(mask, mask, {}, {-1, -1}, 2);
    cv::dilate(mask, mask, {}, {-1, -1}, 2);
    cv::imwrite("step2_table_mask.jpg", mask);
}

// Step 3
void extractTableCorners(const cv::Mat &binaryMask, cv::Mat &visualOutput)
{
    vector<vector<cv::Point>> contours;
    vector<cv::Point> approx, hull;
    cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0.0;
    int maxIdx = -1;
    for (int i = 0; i < contours.size(); ++i)
    {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea)
        {
            maxArea = area;
            maxIdx = i;
        }
    }
    if (maxIdx == -1)
        return;

    cv::approxPolyDP(contours[maxIdx], approx, 20, true);
    cv::convexHull(approx, hull);
    tableCorners = hull;

    visualOutput = img.clone();
    for (size_t i = 0; i < hull.size(); ++i)
    {
        cv::circle(visualOutput, hull[i], 8, cv::Scalar(0, 0, 255), -1);
        cv::putText(visualOutput, to_string(i), hull[i] + cv::Point(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
    cv::drawContours(visualOutput, contours, maxIdx, cv::Scalar(255, 0, 0), 2);
    cv::imwrite("step3_corners.jpg", visualOutput);
}

// Step 4-1: 일반 HSV + Hough 방식 (흰색/노란색 공용)
vector<cv::Point> findBallCenters_HSV_Hough(cv::Scalar lower, cv::Scalar upper, const string &colorName)
{
    vector<cv::Point> centers;

    cv::Mat mask, masked;
    cv::inRange(hsvImage, lower, upper, mask);
    cv::bitwise_and(img, img, masked, mask);

    cv::imwrite("debug_mask_" + colorName + ".jpg", mask);
    cv::imwrite("debug_masked_" + colorName + ".jpg", masked);

    cv::Mat gray;
    cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2);

    vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                     20, 100, 20, 10, 30);

    for (const auto &c : circles)
    {
        cv::Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        centers.push_back(center);
        cout << "[Hough] " << colorName << " ball center: (" << center.x << ", " << center.y << "), r=" << radius << endl;
    }

    return centers;
}

// Step 4-2: 빨간 공 전용 (Hue 0~10 + 170~180)
vector<cv::Point> findBallCenters_HSV_Hough_Red(const string &colorName)
{
    vector<cv::Point> centers;

    cv::Mat mask1, mask2, mask, masked;
    cv::inRange(hsvImage, cv::Scalar(0, 70, 70), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsvImage, cv::Scalar(170, 70, 70), cv::Scalar(180, 255, 255), mask2);
    cv::bitwise_or(mask1, mask2, mask);

    cv::bitwise_and(img, img, masked, mask);
    cv::imwrite("debug_mask_" + colorName + ".jpg", mask);
    cv::imwrite("debug_masked_" + colorName + ".jpg", masked);

    cv::Mat gray;
    cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2);

    vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                     20, 100, 20, 10, 30);

    for (const auto &c : circles)
    {
        cv::Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        centers.push_back(center);
        cout << "[Hough] " << colorName << " ball center: (" << center.x << ", " << center.y << "), r=" << radius << endl;
    }

    return centers;
}

// 픽셀 → mm 변환
cv::Point2f convertToMM(const cv::Point &p, const vector<cv::Point> &corners)
{
    if (corners.size() != 4)
        return {-1, -1};
    float w_mm = 2448.0f, h_mm = 1224.0f;
    float w_px = cv::norm(corners[0] - corners[1]);
    float h_px = cv::norm(corners[0] - corners[3]);
    float scaleX = w_mm / w_px;
    float scaleY = h_mm / h_px;
    float dx = static_cast<float>(p.x - corners[0].x);
    float dy = static_cast<float>(p.y - corners[0].y);
    return {dx * scaleX, dy * scaleY};
}

// 시각화
void drawBallCenters(const vector<cv::Point> &centers, const cv::Scalar &color, const string &label, cv::Mat &output)
{
    for (size_t i = 0; i < centers.size(); ++i)
    {
        cv::circle(output, centers[i], 10, color, 2);
        cv::putText(output, label + to_string(i + 1), centers[i] + cv::Point(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

// main
int main()
{
    if (!loadAndConvertImage("BilliardImage.jpg"))
    {
        cerr << "Image load failed." << endl;
        return -1;
    }

    maskTableArea();

    cv::Mat step2Mask = cv::imread("step2_table_mask.jpg", cv::IMREAD_GRAYSCALE);
    if (step2Mask.empty())
        return -1;

    cv::Mat step3Output;
    extractTableCorners(step2Mask, step3Output);

    // 공 인식 (Red: 두 Hue 범위 합침)
    auto redCenters = findBallCenters_HSV_Hough_Red("red");
    auto whiteCenters = findBallCenters_HSV_Hough(cv::Scalar(0, 0, 180), cv::Scalar(180, 60, 255), "white");
    auto orangeCenters = findBallCenters_HSV_Hough(cv::Scalar(20, 100, 150), cv::Scalar(35, 255, 255), "orange");

    for (const auto &pt : redCenters)
    {
        cv::Point2f mm = convertToMM(pt, tableCorners);
        cout << "Red ball mm pos: (" << mm.x << ", " << mm.y << ")" << endl;
    }

    // 결과 시각화
    cv::Mat step4Output = img.clone();
    drawBallCenters(redCenters, cv::Scalar(0, 0, 255), "R", step4Output);
    drawBallCenters(whiteCenters, cv::Scalar(255, 255, 255), "W", step4Output);
    drawBallCenters(orangeCenters, cv::Scalar(0, 165, 255), "O", step4Output);

    cv::imshow("Step 4 - Ball Centers", step4Output);
    cv::imwrite("step4_result.jpg", step4Output);

    cv::waitKey(0);
    return 0;
}
