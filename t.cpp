#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;
using namespace cv;

// ================================
// 표준 규격 상수 (mm 단위)
// ================================
static constexpr double kTABLE_WIDTH_MM = 2840.0;  // 가로(긴 변)
static constexpr double kTABLE_HEIGHT_MM = 1420.0; // 세로(짧은 변)
static constexpr double kBALL_DIAMETER_MM = 61.5;  // 표준 당구공 지름
static constexpr double kBALL_RADIUS_MM = kBALL_DIAMETER_MM * 0.5;

// ================================
// 유틸: 점 4개를 TL, TR, BL, BR 순서로 정렬
//  - 입력: 임의 순서의 사각형 꼭짓점 4개
//  - 출력: TL, TR, BL, BR
// ================================
static vector<Point2f> orderCornersTLTRBLBR(const vector<Point2f> &pts)
{
    CV_Assert(pts.size() == 4);
    vector<Point2f> out(4);
    // 좌표 합/차를 이용해 정렬
    // TL: (x+y) 최소, BR: (x+y) 최대
    // TR: (x-y) 최대, BL: (x-y) 최소
    auto sumCmp = [](const Point2f &p1, const Point2f &p2)
    { return (p1.x + p1.y) < (p2.x + p2.y); };
    auto diffCmp = [](const Point2f &p1, const Point2f &p2)
    { return (p1.x - p1.y) < (p2.x - p2.y); };

    auto tl = *min_element(pts.begin(), pts.end(), sumCmp);
    auto br = *max_element(pts.begin(), pts.end(), sumCmp);
    auto bl = *min_element(pts.begin(), pts.end(), diffCmp);
    auto tr = *max_element(pts.begin(), pts.end(), diffCmp);

    // 간혹 tl/bl, tr/br이 겹치는 경우가 있어 보정
    // 사분면 근사로 한번 더 걸러줌
    // 중심을 기준으로 사분면 재배치
    Point2f c(0, 0);
    for (auto &p : pts)
        c += p;
    c *= (1.0f / 4.0f);

    vector<pair<Point2f, int>> quads; // (pt, idx)
    for (const auto &p : pts)
    {
        int q = (p.x < c.x ? 0 : 1) + (p.y < c.y ? 0 : 2); // 0:TL,1:TR,2:BL,3:BR 대략
        quads.push_back({p, q});
    }
    // 가장 가까운 사분면에 배치
    // 사분면별 후보가 여러개면 거리로 선택
    auto pick = [&](int want) -> Point2f
    {
        double bestD = 1e18;
        Point2f bestP;
        for (auto &pr : quads)
        {
            if (pr.second == want)
            {
                double d = norm(pr.first - c);
                if (d < bestD)
                {
                    bestD = d;
                    bestP = pr.first;
                }
            }
        }
        // 해당 사분면에 후보가 없으면 sum/diff 기반으로 대체
        if (bestD > 1e17)
        {
            if (want == 0)
                return tl;
            if (want == 1)
                return tr;
            if (want == 2)
                return bl;
            return br;
        }
        return bestP;
    };

    out[0] = pick(0); // TL
    out[1] = pick(1); // TR
    out[2] = pick(2); // BL
    out[3] = pick(3); // BR
    return out;
}

// ================================
// 함수: 당구대(파란 천) 마스크 생성
//  - HSV로 파란색 영역을 추출하고, 모폴로지로 정제
// ================================
static Mat makeBlueTableMask(const Mat &bgr)
{
    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);

    // 파란 당구대 천 범위(환경 따라 조정 필요)
    Scalar lowerBlue(90, 80, 60);
    Scalar upperBlue(140, 255, 255);

    Mat mask;
    inRange(hsv, lowerBlue, upperBlue, mask);

    // 모폴로지 연산으로 잡티 제거 및 구멍 메움
    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, k, Point(-1, -1), 2);
    morphologyEx(mask, mask, MORPH_OPEN, k, Point(-1, -1), 1);

    // 선택적으로 팽창하여 쿠션 안쪽 경계까지 조금 더 포함할 수 있음
    // dilate(mask, mask, k, Point(-1,-1), 1);

    return mask;
}

// ================================
// 함수: 당구대 내측 4꼭짓점 자동 검출
//  - 전략: 파란 천 마스크의 최대 컨투어 → 근사 사각형(approxPolyDP)
//  - 실패 시: 회전사각형(minAreaRect)로 대체 후, 4점 복원
//  - 출력은 TL, TR, BL, BR 순서
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

    // 가장 큰 컨투어 선택
    size_t bestIdx = 0;
    double bestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double area = contourArea(contours[i]);
        if (area > bestArea)
        {
            bestArea = area;
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
            pts.push_back(Point2f((float)p.x, (float)p.y));
    }
    else
    {
        // 회전사각형으로 대체
        RotatedRect rr = minAreaRect(contours[bestIdx]);
        Point2f rrPts[4];
        rr.points(rrPts);
        for (int i = 0; i < 4; ++i)
            pts.push_back(rrPts[i]);
    }

    // 순서 정렬
    corners = orderCornersTLTRBLBR(pts);
    return true;
}

// ================================
// 함수: HSV 범위로 공의 중심점을 찾음 (질문자가 준 코드 확장)
//  - 빨강(2구역) 처리 가능
//  - 너무 작은 잡음 제거, 모폴로지 정제
// ================================
static vector<Point2f> findBallCenters(
    const Mat &hsv_image,
    const Scalar &lower1, const Scalar &upper1,
    const Scalar &lower2 = Scalar(-1, -1, -1), const Scalar &upper2 = Scalar(-1, -1, -1),
    int minArea = 100, int erodeIter = 1, int dilateIter = 3)
{
    Mat mask1, mask2, mask;
    inRange(hsv_image, lower1, upper1, mask1);

    if (lower2[0] != -1) // H wrap-around (빨강)
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
// 결과 묶음 구조체
// ================================
struct BallResult
{
    vector<Point2f> red;    // 정규 스크린 좌표계상의 위치
    vector<Point2f> yellow; // 정규 스크린 좌표계상의 위치
    vector<Point2f> white;  // 정규 스크린 좌표계상의 위치
    Mat canvas;             // 파란 배경의 정규 스크린 이미지 (공이 그려진 결과)
};

// ================================
// 함수: 전체 파이프라인
//  - 입력: 원본 BGR 이미지, 출력 스크린 해상도(픽셀)
//  - 동작:
//    1) 당구대 4점 자동 검출
//    2) Homography로 정규 스크린 좌표계(W×H) 생성
//    3) HSV로 공 색상별 중심 좌표(원본) 검출
//    4) 중심 좌표를 정규 스크린 좌표로 투영
//    5) 표준 공 크기(mm)를 픽셀로 변환하여 그려 넣음
//  - 반환: 색상별 좌표 + 캔버스 이미지
// ================================
static BallResult processBilliardFrame(
    const Mat &bgr,
    int outWidthPx = 1200,
    int outHeightPx = 600,
    bool drawOutlines = true,
    Mat *dbgMask = nullptr)
{
    BallResult res;

    // 1) 당구대 4점 검출
    vector<Point2f> corners;
    bool ok = detectTableInnerCorners(bgr, corners, dbgMask);
    if (!ok)
    {
        cerr << "[Error] 당구대 4점 검출 실패" << endl;
        return res;
    }

    // 2) 정규 스크린 좌표계로의 Homography
    vector<Point2f> dstPts = {
        {0.f, 0.f},                             // TL
        {(float)outWidthPx, 0.f},               // TR
        {0.f, (float)outHeightPx},              // BL
        {(float)outWidthPx, (float)outHeightPx} // BR
    };
    Mat H = findHomography(corners, dstPts);
    if (H.empty())
    {
        cerr << "[Error] Homography 계산 실패" << endl;
        return res;
    }

    // 3) 공 색상별 중심 검출 (원본 좌표계)
    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);

    // 빨강 (Hue 래핑)
    Scalar red_l1(0, 120, 70), red_u1(10, 255, 255);
    Scalar red_l2(170, 120, 70), red_u2(179, 255, 255);

    // 노랑
    Scalar yellow_l(20, 100, 100), yellow_u(30, 255, 255);

    // 흰색 (조명 따라 V/S 조금 풀어야 할 수 있음)
    Scalar white_l(0, 0, 180), white_u(179, 60, 255);

    vector<Point2f> red_c = findBallCenters(hsv, red_l1, red_u1, red_l2, red_u2, /*minArea=*/80);
    vector<Point2f> yellow_c = findBallCenters(hsv, yellow_l, yellow_u, Scalar(-1, -1, -1), Scalar(-1, -1, -1), /*minArea=*/80);
    vector<Point2f> white_c = findBallCenters(hsv, white_l, white_u, Scalar(-1, -1, -1), Scalar(-1, -1, -1), /*minArea=*/80);

    // 4) 좌표를 정규 스크린 좌표계로 변환
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

    // 5) 정규 스크린 캔버스 생성 및 공 그리기
    //    - mm → pixel 스케일
    const double px_per_mm_x = (double)outWidthPx / kTABLE_WIDTH_MM;
    const double px_per_mm_y = (double)outHeightPx / kTABLE_HEIGHT_MM;
    // 이상적으로 두 값은 동일(2:1 비율 정확히 유지), 혹시 오차 있으면 평균 사용
    const double px_per_mm = 0.5 * (px_per_mm_x + px_per_mm_y);

    const int ball_radius_px = (int)std::round(kBALL_RADIUS_MM * px_per_mm);

    // 파란 배경(스크린)
    // BGR: 약간 어두운 당구대 청록/파랑 느낌
    Scalar feltColor(180, 120, 30); // (B,G,R) = 파란/청록 톤, 필요시 조정
    res.canvas = Mat(outWidthPx > 0 && outHeightPx > 0 ? Size(outWidthPx, outHeightPx) : Size(1200, 600),
                     CV_8UC3, feltColor);

    // Anti-aliased로 공을 채워서 그리기
    auto drawBalls = [&](const vector<Point2f> &centers, const Scalar &bgrColor)
    {
        for (const auto &p : centers)
        {
            circle(res.canvas, p, ball_radius_px, bgrColor, FILLED, LINE_AA);
            if (drawOutlines)
                circle(res.canvas, p, ball_radius_px, Scalar(0, 0, 0), 2, LINE_AA);
        }
    };

    drawBalls(res.red, Scalar(0, 0, 255));       // 빨강 (BGR)
    drawBalls(res.yellow, Scalar(0, 255, 255));  // 노랑
    drawBalls(res.white, Scalar(255, 255, 255)); // 흰색

    return res;
}

// ================================
// 데모용 main
//  - 역할:
//    1) 이미지 로드
//    2) 전체 파이프라인 호출
//    3) 색상별 좌표 출력 (정규 스크린 픽셀 좌표계)
//    4) 결과 이미지 표시/저장
// ================================
int main()
{
    // 1) 원본 이미지 로드
    Mat src = imread("1619.png");
    if (src.empty())
    {
        cerr << "[Error] 이미지 로드 실패: 1619.png" << endl;
        return -1;
    }

    // 2) 처리 (출력 스크린 해상도는 자유롭게 조절 가능)
    Mat dbgMask;
    BallResult result = processBilliardFrame(src,
                                             /*outWidthPx=*/1200,
                                             /*outHeightPx=*/600,
                                             /*drawOutlines=*/true,
                                             /*dbgMask=*/&dbgMask);

    // 3) 좌표 출력
    auto printPoints = [](const string &name, const vector<Point2f> &v)
    {
        cout << name << " (" << v.size() << "):" << endl;
        for (size_t i = 0; i < v.size(); ++i)
            cout << "  [" << i << "] (" << v[i].x << ", " << v[i].y << ")\n";
    };
    printPoints("Red", result.red);
    printPoints("Yellow", result.yellow);
    printPoints("White", result.white);

    // 4) 결과 표시 및 저장
    if (!result.canvas.empty())
    {
        imshow("Debug: Blue Table Mask", dbgMask);
        imshow("Original", src);
        imshow("Standard Screen (Balls Rendered)", result.canvas);
        imwrite("standard_screen_result.png", result.canvas);
        cout << "Saved: standard_screen_result.png" << endl;
        waitKey(0);
    }
    else
    {
        cerr << "[Warn] 결과 캔버스가 비어 있음" << endl;
    }

    return 0;
}
