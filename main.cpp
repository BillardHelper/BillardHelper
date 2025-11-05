#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

// ================================
// 표준 규격 상수 (mm 단위)
// ================================
static constexpr double kTABLE_WIDTH_MM = 2540.0;  // "긴 변" (국제식 캐롬 플레이영역 가로)
static constexpr double kTABLE_HEIGHT_MM = 1270.0; // "짧은 변" (국제식 캐롬 플레이영역 세로)
static constexpr double kBALL_DIAMETER_MM = 61.5;  // 표준 당구공 지름
static constexpr double kBALL_RADIUS_MM = kBALL_DIAMETER_MM * 0.5;

// ================================
// 유틸: 점 4개를 TL, TR, BL, BR 순서로 정렬
// ================================
static vector<Point2f> orderCornersTLTRBLBR(const vector<Point2f> &pts)
{
    CV_Assert(pts.size() == 4);
    vector<Point2f> out(4);

    // 중심점
    Point2f c(0, 0);
    for (auto &p : pts)
        c += p;
    c *= (1.0f / 4.0f);

    // 대략적 사분면 분류
    // 0:TL, 1:TR, 2:BL, 3:BR
    auto quad = [&](const Point2f &p)
    {
        int qx = (p.x < c.x) ? 0 : 1;
        int qy = (p.y < c.y) ? 0 : 2;
        return qx + qy;
    };

    // 초기 배치
    for (auto &p : pts)
    {
        int q = quad(p);
        if (q == 0)
            out[0] = p;
        else if (q == 1)
            out[1] = p;
        else if (q == 2)
            out[2] = p;
        else
            out[3] = p;
    }

    // 보정: x+y 최소=TL, 최대=BR
    auto sumCmp = [](const Point2f &a, const Point2f &b)
    { return (a.x + a.y) < (b.x + b.y); };
    Point2f tl = *min_element(pts.begin(), pts.end(), sumCmp);
    Point2f br = *max_element(pts.begin(), pts.end(), sumCmp);

    // 나머지 두 점을 TR/BL로 배치
    vector<Point2f> rest;
    for (auto &p : pts)
        if (p != tl && p != br)
            rest.push_back(p);
    Point2f tr, bl;
    if (rest.size() == 2)
    {
        if (rest[0].x > rest[1].x)
        {
            tr = rest[0];
            bl = rest[1];
        }
        else
        {
            tr = rest[1];
            bl = rest[0];
        }
    }
    else
    {
        // fallback
        tr = out[1];
        bl = out[2];
    }

    out[0] = tl;
    out[1] = tr;
    out[2] = bl;
    out[3] = br;
    return out;
}

// ================================
// 파란 천 마스크 (윤곽 추출용)
// ================================
static Mat makeBlueTableMask(const Mat &bgr)
{
    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);

    // 파란 천 범위(환경에 맞게 조정)
    Scalar lowerBlue(90, 80, 60);
    Scalar upperBlue(140, 255, 255);

    Mat mask;
    inRange(hsv, lowerBlue, upperBlue, mask);

    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, k, Point(-1, -1), 2);
    morphologyEx(mask, mask, MORPH_OPEN, k, Point(-1, -1), 1);
    return mask;
}

// ================================
// 당구대 내측 4꼭짓점 자동 검출 (TL,TR,BL,BR)
// ================================
static bool detectTableInnerCorners(const Mat &bgr, vector<Point2f> &corners, Mat *dbgMask = nullptr)
{
    Mat mask = makeBlueTableMask(bgr);
    if (dbgMask)
        *dbgMask = mask.clone();

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        return false;

    // 가장 큰 컨투어
    size_t bestIdx = 0;
    double bestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double a = contourArea(contours[i]);
        if (a > bestArea)
        {
            bestArea = a;
            bestIdx = i;
        }
    }

    // 다각형 근사
    vector<Point> approx;
    double peri = arcLength(contours[bestIdx], true);
    approxPolyDP(contours[bestIdx], approx, 0.02 * peri, true);

    vector<Point2f> pts;
    if (approx.size() == 4)
    {
        for (auto &p : approx)
            pts.emplace_back((float)p.x, (float)p.y);
    }
    else
    {
        RotatedRect rr = minAreaRect(contours[bestIdx]);
        Point2f rrPts[4];
        rr.points(rrPts);
        for (int i = 0; i < 4; ++i)
            pts.push_back(rrPts[i]);
    }

    corners = orderCornersTLTRBLBR(pts);
    return true;
}

// ================================
// HSV 범위로 공 중심 검출
// ================================
static vector<Point2f> findBallCenters(
    const Mat &hsv_image,
    const Scalar &lower1, const Scalar &upper1,
    const Scalar &lower2 = Scalar(-1, -1, -1), const Scalar &upper2 = Scalar(-1, -1, -1),
    int minArea = 100, int erodeIter = 1, int dilateIter = 3)
{
    Mat mask1, mask2, mask;
    inRange(hsv_image, lower1, upper1, mask1);
    if (lower2[0] != -1)
    {
        inRange(hsv_image, lower2, upper2, mask2);
        bitwise_or(mask1, mask2, mask);
    }
    else
    {
        mask = mask1;
    }

    erode(mask, mask, Mat(), Point(-1, -1), erodeIter);
    dilate(mask, mask, Mat(), Point(-1, -1), dilateIter);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point2f> centers;
    for (const auto &c : contours)
    {
        if (contourArea(c) < minArea)
            continue;
        Moments m = moments(c, true);
        if (m.m00 > 0.0)
            centers.emplace_back((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
    }
    return centers;
}

// ================================
// 결과 구조체
// ================================
struct BallResult
{
    vector<Point2f> red;
    vector<Point2f> yellow;
    vector<Point2f> white;
    Mat canvas; // 표준 스크린에 공이 그려진 결과
};

// ================================
// 전체 파이프라인
//  - 방향(가로/세로) 자동 판정 포함
// ================================
static BallResult processBilliardFrame(
    const Mat &bgr,
    bool drawOutlines = true,
    Mat *dbgMask = nullptr)
{
    BallResult res;

    // 1) 4점 검출
    vector<Point2f> corners;
    if (!detectTableInnerCorners(bgr, corners, dbgMask))
    {
        cerr << "[Error] 당구대 4점 검출 실패" << endl;
        return res;
    }

    // 2) 방향(가로/세로) 판정
    //    topLen: TL-TR, leftLen: TL-BL
    double topLen = norm(corners[1] - corners[0]);
    double leftLen = norm(corners[2] - corners[0]);
    bool isTall = (leftLen > topLen); // 세로형(긴변↑)이면 true

    // 3) 스크린 크기 및 물리 길이 매핑
    int outW, outH;
    double physX_mm, physY_mm; // x축 px이 대응하는 물리 길이, y축 px이 대응하는 물리 길이
    if (!isTall)
    {
        // 가로형: 긴변이 x
        outW = 1200;
        outH = 600;
        physX_mm = kTABLE_WIDTH_MM;  // 2840mm
        physY_mm = kTABLE_HEIGHT_MM; // 1420mm
    }
    else
    {
        // 세로형: 긴변이 y
        outW = 600;
        outH = 1200;
        physX_mm = kTABLE_HEIGHT_MM; // 1420mm (x는 짧은 변)
        physY_mm = kTABLE_WIDTH_MM;  // 2840mm (y는 긴 변)
    }

    // 4) Homography (순서 보존: TL→(0,0), TR→(W,0), BL→(0,H), BR→(W,H))
    vector<Point2f> dstPts = {
        {0.f, 0.f},
        {(float)outW, 0.f},
        {0.f, (float)outH},
        {(float)outW, (float)outH}};
    Mat H = findHomography(corners, dstPts);
    if (H.empty())
    {
        cerr << "[Error] Homography 계산 실패" << endl;
        return res;
    }

    // 5) 공 중심 검출 (원본 좌표계)
    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Scalar red_l1(0, 120, 70), red_u1(10, 255, 255);
    Scalar red_l2(170, 120, 70), red_u2(179, 255, 255);
    Scalar yellow_l(20, 100, 100), yellow_u(30, 255, 255);
    Scalar white_l(0, 0, 180), white_u(179, 60, 255);

    vector<Point2f> red_c = findBallCenters(hsv, red_l1, red_u1, red_l2, red_u2, 80);
    vector<Point2f> yellow_c = findBallCenters(hsv, yellow_l, yellow_u, Scalar(-1, -1, -1), Scalar(-1, -1, -1), 80);
    vector<Point2f> white_c = findBallCenters(hsv, white_l, white_u, Scalar(-1, -1, -1), Scalar(-1, -1, -1), 80);

    // 6) 좌표를 정규 스크린 좌표계로 변환
    vector<Point2f> red_t, yellow_t, white_t;
    if (!red_c.empty())
        perspectiveTransform(red_c, red_t, H);
    if (!yellow_c.empty())
        perspectiveTransform(yellow_c, yellow_t, H);
    if (!white_c.empty())
        perspectiveTransform(white_c, white_t, H);

    res.red = red_t;
    res.yellow = yellow_t;
    res.white = white_t;

    // 7) mm → px 스케일(방향에 맞춰 정확 매핑)
    const double px_per_mm_x = (double)outW / physX_mm;
    const double px_per_mm_y = (double)outH / physY_mm;
    const double px_per_mm = 0.5 * (px_per_mm_x + px_per_mm_y); // 평균 사용
    const int ball_radius_px = (int)std::round(kBALL_RADIUS_MM * px_per_mm);

    // 8) 파란 배경 스크린 생성 및 공 그리기
    Scalar feltColor(180, 120, 30); // BGR 청록/파랑 느낌
    res.canvas = Mat(Size(outW, outH), CV_8UC3, feltColor);

    auto drawBalls = [&](const vector<Point2f> &centers, const Scalar &bgrColor)
    {
        for (const auto &p : centers)
        {
            circle(res.canvas, p, ball_radius_px, bgrColor, FILLED, LINE_AA);
            if (drawOutlines)
                circle(res.canvas, p, ball_radius_px, Scalar(0, 0, 0), 2, LINE_AA);
        }
    };
    drawBalls(res.red, Scalar(0, 0, 255));       // 빨강
    drawBalls(res.yellow, Scalar(0, 255, 255));  // 노랑
    drawBalls(res.white, Scalar(255, 255, 255)); // 흰색

    return res;
}

// ================================
// 데모용 main
// ================================
int main()
{
    // 원본 이미지: x가 짧은 변, y가 긴 변으로 찍힌 상태
    Mat src = imread("TableAndBallsOnly.png");
    if (src.empty())
    {
        cerr << "[Error] 이미지 로드 실패" << endl;
        return -1;
    }

    Mat dbgMask;
    BallResult result = processBilliardFrame(src,
                                             /*drawOutlines=*/true,
                                             /*dbgMask=*/&dbgMask);

    auto printPoints = [](const string &name, const vector<Point2f> &v)
    {
        cout << name << " (" << v.size() << "):\n";
        for (size_t i = 0; i < v.size(); ++i)
            cout << "  [" << i << "] (" << v[i].x << ", " << v[i].y << ")\n";
    };
    printPoints("Red", result.red);
    printPoints("Yellow", result.yellow);
    printPoints("White", result.white);

    if (!result.canvas.empty())
    {
        imshow("Debug: Blue Table Mask", dbgMask);
        imshow("Original", src);
        imshow("Standard Screen (Balls Rendered)", result.canvas);
        imwrite("standard_screen_result.png", result.canvas);
        cout << "Saved: standard_screen_result.png\n";
        waitKey(0);
    }
    else
    {
        cerr << "[Warn] 결과 캔버스가 비어 있음\n";
    }
    return 0;
}
