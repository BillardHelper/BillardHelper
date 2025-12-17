#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

// ------------------------------------------------------------
// 조명(밝기) 보정: HSV V 채널 equalizeHist
// ------------------------------------------------------------
static Mat LightingNormalizeHSV(const Mat &srcBGR)
{
    if (srcBGR.empty() || srcBGR.type() != CV_8UC3)
    {
        cerr << "[LightingNormalizeHSV] invalid input image" << endl;
        return srcBGR.clone();
    }

    Mat hsv;
    cvtColor(srcBGR, hsv, COLOR_BGR2HSV);

    vector<Mat> ch;
    split(hsv, ch); // ch[0]=H, ch[1]=S, ch[2]=V

    equalizeHist(ch[2], ch[2]); // 밝기 균일화

    Mat hsv_eq;
    merge(ch, hsv_eq);

    Mat outBGR;
    cvtColor(hsv_eq, outBGR, COLOR_HSV2BGR);

    return outBGR;
}

// ------------------------------------------------------------
// 테이블 내부 전체를 255로 만드는 마스크 생성 함수 (수정본)
//  - 파란 천 HSV로 "윤곽"만 추출
//  - 최대 컨투어 -> 근사 다각형/볼록껍질 -> 내부를 통째로 채움
//  - 필요 시 border_shrink_px로 살짝 erode하여 경계 누수 방지
// ------------------------------------------------------------
static Mat MakeTableMask(const Mat &srcBGR, int border_shrink_px = 0)
{
    // 1) BGR -> HSV
    Mat hsv;
    cvtColor(srcBGR, hsv, COLOR_BGR2HSV);

    // 2) 파란 천 범위 (환경에 맞게 S/V 조정 가능)
    Scalar lowerBlue(90, 80, 60);
    Scalar upperBlue(140, 255, 255);
    Mat maskBlue;
    inRange(hsv, lowerBlue, upperBlue, maskBlue);

    // 3) 노이즈 제거 (close -> open)
    Mat k5 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(maskBlue, maskBlue, MORPH_CLOSE, k5);
    morphologyEx(maskBlue, maskBlue, MORPH_OPEN, k5);

    // 4) 최대 컨투어 찾기
    vector<vector<Point>> contours;
    findContours(maskBlue, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat tableMask = Mat::zeros(maskBlue.size(), CV_8UC1);
    if (contours.empty())
        return tableMask; // 비어 있으면 그대로 반환

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

    // 5) 근사 다각형 + 볼록껍질로 안정화
    vector<Point> approx;
    double peri = arcLength(contours[bestIdx], true);
    approxPolyDP(contours[bestIdx], approx, 0.02 * peri, true);

    // 근사 결과가 불안정하면 볼록껍질로 대체
    if (approx.size() < 3)
        approx = contours[bestIdx];

    vector<Point> hull;
    convexHull(approx, hull);

    if (hull.size() >= 3)
    {
        const vector<vector<Point>> fillMe{hull};
        fillPoly(tableMask, fillMe, Scalar(255));
    }

    // 6) 경계가 살짝 새면 안쪽으로 줄이기(옵션)
    if (border_shrink_px > 0)
    {
        Mat k = getStructuringElement(MORPH_ELLIPSE,
                                      Size(2 * border_shrink_px + 1, 2 * border_shrink_px + 1));
        erode(tableMask, tableMask, k, Point(-1, -1), 1);
    }

    // 결과: 테이블 내부는 전부 255, 외부는 0
    return tableMask;
}

// ------------------------------------------------------------
// 테이블(파란 천) 마스크 만들기
//    - 파란 천 Hue 범위로 inRange
//    - Morphology로 다듬고
//    - 약간 dilate해서 쿠션 경계까지 포함
//    반환: mask (CV_8UC1, 0 또는 255)
// ------------------------------------------------------------
/*
static Mat MakeTableMask(const Mat &srcBGR)
{
    // 1. BGR -> HSV
    Mat hsv;
    cvtColor(srcBGR, hsv, COLOR_BGR2HSV);

    // 2. 파란 당구대 천만 먼저 threshold
    Scalar lowerBlue(90, 80, 60);    // H,S,V 최소
    Scalar upperBlue(140, 255, 255); // H,S,V 최대
    Mat maskBlue;
    inRange(hsv, lowerBlue, upperBlue, maskBlue);

    // 3. 노이즈 제거 (close -> open)
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(maskBlue, maskBlue, MORPH_CLOSE, kernel);
    morphologyEx(maskBlue, maskBlue, MORPH_OPEN, kernel);

    // ---------------------------
    // (A) 테이블 "윤곽 전체" 마스크 만들기
    // ---------------------------
    // 아이디어:
    //  - maskBlue에서 가장 큰 컨투어(=당구대 영역) 찾기
    //  - 컨투어를 근사 다각형(거의 사각형)으로 만들고
    //  - 다각형으로 꽉 채운 마스크를 tableHullMask로 쓴다
    //
    // 이렇게 만든 tableHullMask는
    //  "테이블 내부 전체"만 255인 마스크가 된다.
    // ---------------------------

    // 컨투어 탐색 준비
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(maskBlue, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat tableHullMask = Mat::zeros(maskBlue.size(), CV_8UC1);

    if (!contours.empty())
    {
        // 가장 큰 컨투어 선택 (면적 최대)
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

        // 선택된 컨투어 근사(사각형 비슷하게) -> 테이블 전체 영역 얻기
        vector<Point> poly;
        approxPolyDP(contours[bestIdx], poly, 10.0, true);
        // poly가 4점 근처일 가능성이 높음 (테이블 사각형 투시된 형태)

        // poly 로 채운 마스크 생성
        vector<vector<Point>> fillMe;
        fillMe.push_back(poly);
        fillPoly(tableHullMask, fillMe, Scalar(255));
    }

    // tableHullMask 가 이제 "테이블 전체" 영역 (255=테이블 내부)

    // ---------------------------
    // 파란 천 마스크를 팽창해서 공까지 포함
    // ---------------------------
    Mat maskDilated;
    dilate(maskBlue, maskDilated, kernel, Point(-1, -1), 2);
    // 여기서 maskDilated는 바깥으로도 조금 새어나갈 수 있음

    // ---------------------------
    // 최종 마스크 = (팽창된 마스크) AND (테이블 전체 영역)
    // ---------------------------
    Mat finalMask;
    bitwise_and(maskDilated, tableHullMask, finalMask);

    // finalMask:
    // - 테이블 영역 밖은 절대 0 (tableHullMask 덕분)
    // - 테이블 안은 원래 파란 천 + 약간 확장 (공 포함)

    return finalMask;
}*/

/*
// dilate 세부 조정 필요(위 코드로 해결)
static Mat MakeTableMask(const Mat &srcBGR)
{
    Mat hsv;
    cvtColor(srcBGR, hsv, COLOR_BGR2HSV);

    // 파란 당구대 천 범위 (넉넉하게 설정)
    Scalar lowerBlue(90, 80, 60);    // H,S,V 최소
    Scalar upperBlue(140, 255, 255); // H,S,V 최대

    Mat maskBlue;
    inRange(hsv, lowerBlue, upperBlue, maskBlue);

    // 잡티 제거 & 구멍 메우기
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(maskBlue, maskBlue, MORPH_CLOSE, kernel);
    morphologyEx(maskBlue, maskBlue, MORPH_OPEN, kernel);

    // 테이블 경계(쿠션 부분)까지 조금 더 포함시키고 싶으면 팽창(dilate), 팽창 안하면, 영역안에 공이 안잡힘, 팽창 하면 공은 잡히는데 당구대 영역이 좀 더 커짐
    // dilate(maskBlue, maskBlue, kernel, Point(-1, -1), 2);

    return maskBlue;
}
*/

// ------------------------------------------------------------
// 최종 합성
//    - tableMask 영역 안: 원본(조명 보정된) 색 유지 (파란 천 + 공 색 전부 그대로)
//    - tableMask 영역 밖: (0,0,0)
// ------------------------------------------------------------
static Mat KeepOnlyInsideTable(const Mat &srcBGR, const Mat &tableMask)
{
    Mat result = Mat::zeros(srcBGR.size(), srcBGR.type());
    srcBGR.copyTo(result, tableMask); // 마스크가 0이 아닌 영역만 복사
    return result;
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(void)
{
    Mat img = imread("test_img1.jpg", IMREAD_COLOR);
    if (img.empty())
    {
        cerr << "Failed to load image: " << endl;
        return 1;
    }

    // 리사이즈(원본 너무 커서 보기 힘듦..)
    int targetHeight = 1000;
    if (img.rows > targetHeight)
    {
        double scale = static_cast<double>(targetHeight) / img.rows;
        resize(img, img, Size(), scale, scale);
        cout << "[Info] resized to " << img.cols << " x " << img.rows << endl;
    }

    // (2) 조명 보정 (밝기 평활화)
    Mat img_norm = LightingNormalizeHSV(img);

    // (3) 테이블 마스크 생성 (파란 천 기반)
    Mat tableMask = MakeTableMask(img_norm);

    // (4) 마스크 안은 모두 살리고 (천 + 공), 마스크 밖은 0으로
    Mat tableAndBallsOnly = KeepOnlyInsideTable(img_norm, tableMask);

    imshow("Original", img);
    imshow("LightingNormalized", img_norm);
    imshow("TableMask", tableMask); // 흰 부분이 테이블로 인식된 영역
    imshow("TableAndBallsOnly", tableAndBallsOnly);

    // 결과 영상 저장
    imwrite("LightingNormalized.png", img_norm);
    imwrite("TableMask.png", tableMask);
    imwrite("TableAndBallsOnly.png", tableAndBallsOnly);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
