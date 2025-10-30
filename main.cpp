// auto_billiard_topdown_minwarp_fixed.cpp
// g++ -std=c++17 auto_billiard_topdown_minwarp_fixed.cpp `pkg-config --cflags --libs opencv4` -o topdown
// 사용: ./topdown test.png
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace cv;

// ====================== 설정 ======================
static const double TABLE_W_M = 2.84; // 긴 변 [m]
static const double TABLE_H_M = 1.42; // 짧은 변 [m]
static const double PPM = 300.0;      // pixels per meter
static const bool DRAW_GRID = true;   // 0.1m 격자 표시

// ====================== 유틸 함수 ======================
static double LL2(const Point2f &a, const Point2f &b)
{
    return hypot(a.x - b.x, a.y - b.y);
}
static double signedArea(const vector<Point2f> &P)
{
    double A = 0;
    int n = (int)P.size();
    for (int i = 0; i < n; i++)
    {
        int j = (i + 1) % n;
        A += P[i].x * P[j].y - P[j].x * P[i].y;
    }
    return 0.5 * A;
}
static double dotn(const Point2f &a, const Point2f &b)
{
    double la = hypot(a.x, a.y), lb = hypot(b.x, b.y);
    if (la < 1e-9 || lb < 1e-9)
        return 1.0;
    return fabs((a.x * b.x + a.y * b.y) / (la * lb));
}

// 4점 아무 순서를 (TL,TR,BR,BL)로 강건히 정렬
static vector<Point2f> orderFourPointsRobust(const vector<Point2f> &pts, double aspect = (2.84 / 1.42))
{
    CV_Assert(pts.size() == 4);
    array<int, 4> idx = {0, 1, 2, 3};
    vector<Point2f> bestOrder(4);
    double bestScore = 1e100;

    do
    {
        vector<Point2f> q = {pts[idx[0]], pts[idx[1]], pts[idx[2]], pts[idx[3]]}; // TL,TR,BR,BL 가정
        double A = signedArea(q);
        if (A <= 0)
            continue; // 반시계만 허용

        double top = LL2(q[0], q[1]), bottom = LL2(q[3], q[2]);
        double left = LL2(q[0], q[3]), right = LL2(q[1], q[2]);
        double meanH = 0.5 * (top + bottom);
        double meanV = 0.5 * (left + right);

        double ratio = (meanV < 1e-6) ? 1e6 : (meanH / meanV);
        double ratio_err = fabs(ratio - aspect);

        double ortho1 = dotn(q[1] - q[0], q[3] - q[0]); // top vs left
        double ortho2 = dotn(q[2] - q[1], q[2] - q[3]); // right vs bottom
        double ortho_err = 0.5 * (ortho1 + ortho2);

        double d1 = LL2(q[0], q[2]), d2 = LL2(q[1], q[3]);
        double diag_err = fabs(d1 - d2) / max(1.0, max(d1, d2));

        double score = 4.0 * ratio_err + 1.0 * ortho_err + 0.5 * diag_err;

        if (score < bestScore)
        {
            bestScore = score;
            bestOrder = q;
        }
    } while (next_permutation(idx.begin(), idx.end()));

    if (bestScore > 1e99)
    {
        // 안전망: 단순 y정렬 -> 상/하, x로 좌/우
        vector<Point2f> p = pts;
        sort(p.begin(), p.end(), [](const Point2f &a, const Point2f &b)
             { return a.y < b.y; });
        vector<Point2f> top2 = {p[0], p[1]}, bot2 = {p[2], p[3]};
        if (top2[0].x > top2[1].x)
            swap(top2[0], top2[1]);
        if (bot2[0].x > bot2[1].x)
            swap(bot2[0], bot2[1]);
        bestOrder = {top2[0], top2[1], bot2[1], bot2[0]};
    }

    // 가로<세로이면 90도 회전 보정(테이블은 가로가 더 김)
    double topL = LL2(bestOrder[0], bestOrder[1]);
    double botL = LL2(bestOrder[3], bestOrder[2]);
    double leftL = LL2(bestOrder[0], bestOrder[3]);
    double rightL = LL2(bestOrder[1], bestOrder[2]);
    double meanH = 0.5 * (topL + botL), meanV = 0.5 * (leftL + rightL);
    if (meanH < meanV)
    {
        vector<Point2f> r = {bestOrder[3], bestOrder[0], bestOrder[1], bestOrder[2]};
        if (signedArea(r) <= 0)
            swap(r[1], r[3]);
        bestOrder = r;
    }
    if (signedArea(bestOrder) <= 0)
        swap(bestOrder[1], bestOrder[3]);
    return bestOrder;
}

// 출력 좌표 두 후보 생성
static void makeTargets(int pxShort, int pxLong,
                        vector<Point2f> &dstA, vector<Point2f> &dstB)
{
    // 후보 A: 가로=짧은, 세로=긴  (짧은 변에서 촬영 시 왜곡이 작은 경우 많음)
    dstA = {
        Point2f(0, 0),
        Point2f((float)pxShort - 1, 0),
        Point2f((float)pxShort - 1, (float)pxLong - 1),
        Point2f(0, (float)pxLong - 1)};
    // 후보 B: 가로=긴, 세로=짧은
    dstB = {
        Point2f(0, 0),
        Point2f((float)pxLong - 1, 0),
        Point2f((float)pxLong - 1, (float)pxShort - 1),
        Point2f(0, (float)pxShort - 1)};
}

// 국소 확대율 최대값 평가(역변환 자코비안의 det 기반)
static double maxMagnification(const Mat &Hinv, int outW, int outH, int grid = 20)
{
    auto mapPt = [&](double x, double y)
    {
        Mat X = (Mat_<double>(3, 1) << x, y, 1.0);
        Mat U = Hinv * X;
        double w = U.at<double>(2, 0);
        return Point2d(U.at<double>(0, 0) / w, U.at<double>(1, 0) / w);
    };
    const double eps = 1.0;
    double worst = 0.0;
    for (int i = 0; i <= grid; i++)
    {
        for (int j = 0; j <= grid; j++)
        {
            double u = (outW - 1) * (double)i / grid;
            double v = (outH - 1) * (double)j / grid;
            Point2d P = mapPt(u, v);
            Point2d Px = mapPt(u + eps, v);
            Point2d Py = mapPt(u, v + eps);
            double j11 = (Px.x - P.x) / eps;
            double j12 = (Py.x - P.x) / eps;
            double j21 = (Px.y - P.y) / eps;
            double j22 = (Py.y - P.y) / eps;
            double detJ = j11 * j22 - j12 * j21;
            double mag = sqrt(fabs(detJ)); // 간단 배율 지표
            worst = max(worst, mag);
        }
    }
    return worst;
}

// 두 후보 중 왜곡(최대 배율)이 작은 후보 선택
static bool chooseBestOrientation(const vector<Point2f> &ordered,
                                  int pxShort, int pxLong,
                                  Mat &H_best, Size &outSize, vector<Point2f> &dstBest)
{
    vector<Point2f> dstA, dstB;
    makeTargets(pxShort, pxLong, dstA, dstB);

    // 후보 A
    Mat H_A = getPerspectiveTransform(ordered, dstA);
    Mat Hinv_A;
    invert(H_A, Hinv_A, DECOMP_LU);
    double M_A = maxMagnification(Hinv_A, pxShort, pxLong);

    // 후보 B
    Mat H_B = getPerspectiveTransform(ordered, dstB);
    Mat Hinv_B;
    invert(H_B, Hinv_B, DECOMP_LU);
    double M_B = maxMagnification(Hinv_B, pxLong, pxShort);

    if (M_A <= M_B)
    {
        H_best = H_A;
        outSize = Size(pxShort, pxLong);
        dstBest = dstA;
        return true;
    }
    else
    {
        H_best = H_B;
        outSize = Size(pxLong, pxShort);
        dstBest = dstB;
        return true;
    }
}

// 0.1m 격자 그리기
static void drawMetricGrid(Mat &img, double ppm, double grid_m = 0.1)
{
    const int W = img.cols, H = img.rows;
    int step = (int)round(grid_m * ppm);
    if (step < 8)
        step = 8; // 너무 촘촘하면 가독성 저하
    for (int x = 0; x < W; x += step)
        line(img, Point(x, 0), Point(x, H - 1), Scalar(180, 180, 180), 1, LINE_AA);
    for (int y = 0; y < H; y += step)
        line(img, Point(0, y), Point(W - 1, y), Scalar(180, 180, 180), 1, LINE_AA);
    for (int x = 0, idx = 0; x < W; x += step, ++idx)
        putText(img, to_string(idx * grid_m) + "m", Point(x + 4, 18),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 50, 50), 1, LINE_AA);
    for (int y = 0, idx = 0; y < H; y += step, ++idx)
        putText(img, to_string(idx * grid_m) + "m", Point(4, y + 18),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 50, 50), 1, LINE_AA);
}

// ====================== 메인 파이프라인 ======================
int main(void)
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Mat src = imread("Test1_TableAndBallsOnly.png", IMREAD_COLOR);
    if (src.empty())
    {
        cerr << "[Error] 이미지 열기 실패. 실행 디렉토리에 파일이 없거나 경로가 잘못됨.\n";
        cerr << "        사용법: ./topdown <image_path>\n";
        cerr << "        종료하려면 Enter 키를 누르세요.\n";
        cin.get(); // 콘솔 즉시 종료 방지(Windows 더블클릭 대비)
        return 1;
    }
    cout << "[Info] src size = " << src.cols << "x" << src.rows << "\n";

    // ---- 전처리: 경계 강화 (배경 검정 가정) ----
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    Mat edges;
    Canny(gray, edges, 50, 150);

    // ---- 가장 큰 컨투어 → 사각 근사 or minAreaRect ----
    vector<vector<Point>> contours;
    vector<Vec4i> hier;
    findContours(edges, contours, hier, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty())
    {
        cerr << "[Error] 컨투어를 찾지 못했음. 전처리(배경 마스크) 확인 필요.\n";
        cerr << "        종료하려면 Enter 키를 누르세요.\n";
        cin.get();
        return 1;
    }
    int bestIdx = -1;
    double bestArea = 0.0;
    for (int i = 0; i < (int)contours.size(); ++i)
    {
        double a = contourArea(contours[i]);
        if (a > bestArea)
        {
            bestArea = a;
            bestIdx = i;
        }
    }
    vector<Point> hull;
    convexHull(contours[bestIdx], hull);

    vector<Point> poly;
    approxPolyDP(hull, poly, 0.01 * arcLength(hull, true), true);

    vector<Point2f> detected;
    if ((int)poly.size() == 4)
    {
        for (auto &p : poly)
            detected.emplace_back((float)p.x, (float)p.y);
        cout << "[Info] approxPolyDP로 사각형(4점) 검출.\n";
    }
    else
    {
        RotatedRect rr = minAreaRect(hull);
        Point2f rrPts[4];
        rr.points(rrPts);
        detected.assign(rrPts, rrPts + 4);
        cout << "[Warn] approxPolyDP가 4점이 아님 → minAreaRect 사용.\n";
    }

    // ---- 강건 정렬 (TL,TR,BR,BL) ----
    vector<Point2f> ordered = orderFourPointsRobust(detected, TABLE_W_M / TABLE_H_M);

    // 디버그: 코너 보기 — polylines는 vector<vector<Point>>를 기대하므로 변환
    Mat dbg = src.clone();
    for (int i = 0; i < 4; ++i)
        circle(dbg, ordered[i], 8, Scalar(0, 255, 255), FILLED, LINE_AA);
    vector<Point> polyDraw;
    for (auto &p : ordered)
        polyDraw.emplace_back(cvRound(p.x), cvRound(p.y));
    vector<vector<Point>> polyList(1, polyDraw);
    polylines(dbg, polyList, true, Scalar(0, 200, 255), 2, LINE_AA);

    // ---- 출력 해상도(메트릭 픽셀) ----
    const int pxLong = (int)round(max(TABLE_W_M, TABLE_H_M) * PPM);
    const int pxShort = (int)round(min(TABLE_W_M, TABLE_H_M) * PPM);
    cout << "[Info] pxLong=" << pxLong << ", pxShort=" << pxShort << ", PPM=" << PPM << "\n";

    // ---- 두 후보 비교 후 왜곡 최소 배치 선택 ----
    Mat H;
    Size outSz;
    vector<Point2f> dstChosen;
    chooseBestOrientation(ordered, pxShort, pxLong, H, outSz, dstChosen);
    cout << "[Info] 선택된 출력크기 = " << outSz.width << " x " << outSz.height << " px\n";

    // ---- 워핑 ----
    Mat top;
    warpPerspective(src, top, H, outSz, INTER_LANCZOS4, BORDER_CONSTANT);

    // ---- 격자 오버레이(선택) ----
    Mat top_grid = top.clone();
    if (DRAW_GRID)
        drawMetricGrid(top_grid, PPM, 0.1);

    // ---- 출력 ----
    imshow("source", src);
    imshow("corners_debug", dbg);
    imshow("topdown", top);
    if (DRAW_GRID)
        imshow("topdown_grid", top_grid);

    imwrite("topdown.png", top);
    if (DRAW_GRID)
        imwrite("topdown_grid.png", top_grid);

    cout << fixed << setprecision(3);
    cout << "[OK] 저장 완료: topdown.png"
         << (DRAW_GRID ? ", topdown_grid.png" : "") << "\n";
    cout << "    코너(TL,TR,BR,BL) = ";
    for (auto &p : ordered)
        cout << "(" << p.x << "," << p.y << ") ";
    cout << "\n";
    cout << "종료하려면 창을 닫거나 아무 키나 누르세요.\n";
    waitKey(0);
    return 0;
}
