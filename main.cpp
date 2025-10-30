// billiard_topdown_reprojection.cpp
// g++ -std=c++17 billiard_topdown_reprojection.cpp `pkg-config --cflags --libs opencv4` -o reproj
// 사용법: ./reproj test.png
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

// -----------------------------
// 설정값
// -----------------------------
// 실측 테이블 크기(미터). 예: 국제식 중대(대략 2.84m x 1.42m)로 가정
static const double TABLE_W_M = 2.84; // 가로(긴 변)
static const double TABLE_H_M = 1.42; // 세로(짧은 변)
// 출력 해상도 스케일: 1m당 픽셀 수 (pixels per meter)
static const double PPM = 300.0; // 300 px/m => 2.84m ≈ 852px, 1.42m ≈ 426px

// 마우스 클릭으로 원본 이미지에서 테이블 네 모서리를 받는다(시계 방향 권장: TL, TR, BR, BL)
static vector<Point2f> g_imgPts;
static Mat g_show;

// 픽셀 → 월드(미터) 변환용 호모그래피(옵션)
static Mat H_img2world_m; // 이미지 픽셀 -> (X_m, Y_m) 세계좌표(미터)로 바로 가는 H

static void onMouse(int event, int x, int y, int, void *)
{
    if (event != EVENT_LBUTTONDOWN)
        return;

    if ((int)g_imgPts.size() < 4)
    {
        g_imgPts.emplace_back((float)x, (float)y);
        // 시각화
        circle(g_show, Point(x, y), 6, Scalar(0, 255, 255), FILLED, LINE_AA);
        putText(g_show, to_string((int)g_imgPts.size()),
                Point(x + 8, y - 8), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2, LINE_AA);
        imshow("select 4 corners (clockwise TL,TR,BR,BL)", g_show);
    }
}

// 보조: 격자(예: 0.1m 간격) 그리기
static void drawMetricGrid(Mat &img, double ppm, double grid_m = 0.1)
{
    const int W = img.cols;
    const int H = img.rows;
    const int step = (int)round(grid_m * ppm);

    // 얇은 회색 라인
    for (int x = 0; x < W; x += step)
        line(img, Point(x, 0), Point(x, H - 1), Scalar(180, 180, 180), 1, LINE_AA);
    for (int y = 0; y < H; y += step)
        line(img, Point(0, y), Point(W - 1, y), Scalar(180, 180, 180), 1, LINE_AA);

    // 축 라벨(미터)
    for (int x = 0, idx = 0; x < W; x += step, ++idx)
        putText(img, to_string(idx * grid_m) + "m", Point(x + 4, 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 50, 50), 1, LINE_AA);
    for (int y = 0, idx = 0; y < H; y += step, ++idx)
        putText(img, to_string(idx * grid_m) + "m", Point(4, y + 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 50, 50), 1, LINE_AA);
}

// 보조: 한 점을 월드(미터) 좌표로 투영
// 입력 pt_img: 원본 이미지 픽셀 좌표
// 반환: (X_m, Y_m) 미터 단위
static Point2f pixelToWorldMeters(const Point2f &pt_img)
{
    vector<Point2f> src(1, pt_img);
    vector<Point2f> dst(1);
    perspectiveTransform(src, dst, H_img2world_m);
    return dst[0];
}

int main(void)
{
    Mat src = imread("Test1_TableAndBallsOnly.png", IMREAD_COLOR);
    if (src.empty())
    {
        cerr << "Failed to read image: " << endl;
        return 1;
    }

    // 1) 코너 선택
    g_show = src.clone();
    namedWindow("select 4 corners (clockwise TL,TR,BR,BL)", WINDOW_NORMAL);
    imshow("select 4 corners (clockwise TL,TR,BR,BL)", g_show);
    setMouseCallback("select 4 corners (clockwise TL,TR,BR,BL)", onMouse);

    cout << "[Info] 원본 이미지에서 테이블 네 모서리를 클릭하세요."
         << " 권장 순서: 좌상(TL) -> 우상(TR) -> 우하(BR) -> 좌하(BL)\n";
    cout << "       클릭 완료 후 아무 키나 누르면 보정이 진행됨.\n";
    waitKey(0);
    destroyWindow("select 4 corners (clockwise TL,TR,BR,BL)");

    if ((int)g_imgPts.size() != 4)
    {
        cerr << "Need 4 points! Clicked: " << g_imgPts.size() << endl;
        return 1;
    }

    // 2) 목적지(탑뷰) 좌표계: 실측(m) → 픽셀 변환으로 직사각형 사각형 정의
    const int outW = (int)round(TABLE_W_M * PPM);
    const int outH = (int)round(TABLE_H_M * PPM);

    // 시계방향으로 목적지 사각형(픽셀 단위)
    vector<Point2f> dstPts_px = {
        Point2f(0.0f, 0.0f),                             // (0,0) m
        Point2f((float)outW - 1.0f, 0.0f),               // (W,0) m
        Point2f((float)outW - 1.0f, (float)outH - 1.0f), // (W,H) m
        Point2f(0.0f, (float)outH - 1.0f)                // (0,H) m
    };

    // 3) Homography: 원본 사각형 -> 메트릭 탑뷰 사각형(픽셀 공간)
    //    getPerspectiveTransform는 4점 대응 전용, findHomography로 대체해도 됨.
    Mat H_src2top = getPerspectiveTransform(g_imgPts, dstPts_px);

    // 4) 워핑(재투영)
    Mat topview;
    warpPerspective(src, topview, H_src2top, Size(outW, outH), INTER_CUBIC, BORDER_CONSTANT);

    // 5) 격자(0.1m) 오버레이(선택)
    Mat gridView = topview.clone();
    drawMetricGrid(gridView, PPM, 0.1);

    // 6) (옵션) 픽셀 → 월드(미터) 변환 행렬 구성
    //    목적: 원본 이미지 좌표에서 (X_m, Y_m)로 바로 변환하고 싶을 때 사용.
    //    방법: 목적지 좌표를 "픽셀"이 아니라 "미터" 좌표계로 직접 두고 H를 만든다.
    vector<Point2f> dstPts_m = {
        Point2f(0.0f, 0.0f),                         // (0,0) m
        Point2f((float)TABLE_W_M, 0.0f),             // (W,0) m
        Point2f((float)TABLE_W_M, (float)TABLE_H_M), // (W,H) m
        Point2f(0.0f, (float)TABLE_H_M)              // (0,H) m
    };
    H_img2world_m = getPerspectiveTransform(g_imgPts, dstPts_m);

    // 샘플: 원본 이미지의 임의의 점을 월드(미터)로 변환해보기
    // 예) 원본 중앙점
    Point2f c_img((float)src.cols / 2.0f, (float)src.rows / 2.0f);
    Point2f c_world_m = pixelToWorldMeters(c_img);
    cout << fixed << setprecision(4);
    cout << "[Debug] src center pixel (" << c_img.x << "," << c_img.y << ") -> world (m): ("
         << c_world_m.x << ", " << c_world_m.y << ")\n";

    // 7) 결과 표시/저장
    imshow("source", src);
    imshow("topview(metric)", topview);
    imshow("topview_with_grid(0.1m)", gridView);
    imwrite("topview.png", topview);
    imwrite("topview_grid.png", gridView);

    cout << "[Done] topview.png, topview_grid.png 저장 완료.\n";
    cout << "       출력 해상도: " << outW << " x " << outH << " px,  PPM=" << PPM << " px/m\n";
    cout << "       이 좌표계는 X->가로(긴변, 0.." << TABLE_W_M << "m), Y->세로(짧은변, 0.." << TABLE_H_M << "m)이다.\n";

    waitKey(0);
    return 0;
}
