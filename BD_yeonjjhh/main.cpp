#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

static const int OUT_W = 1224;
static const int OUT_H = 2448;

struct BallDet
{
    string name;
    Point2f center_warp; // center in warped space (1224x2448)
    Scalar bgr;          // draw color
    bool valid = false;
    double score = 0.0; // quality score
};

// --- This function sorts 4 corner points to (tl, tr, br, bl).
static vector<Point2f> orderCornersTLTRBRBL(const vector<Point2f> &pts4)
{
    if (pts4.size() != 4)
        return {};
    Point2f tl, tr, br, bl;
    double minSum = 1e18, maxSum = -1e18;
    double minDiff = 1e18, maxDiff = -1e18;

    for (auto &pt : pts4)
    {
        double s = pt.x + pt.y;
        double d = pt.x - pt.y;
        if (s < minSum)
        {
            minSum = s;
            tl = pt;
        }
        if (s > maxSum)
        {
            maxSum = s;
            br = pt;
        }
        if (d < minDiff)
        {
            minDiff = d;
            tr = pt;
        }
        if (d > maxDiff)
        {
            maxDiff = d;
            bl = pt;
        }
    }
    return {tl, tr, br, bl};
}

// --- This function finds the largest external contour and returns its convex 4-corner polygon if possible.
static bool findLargestContourApprox4(const Mat &bin, vector<Point> &approxOut)
{
    vector<vector<Point>> contours;
    findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        return false;

    int bestIdx = -1;
    double bestArea = 0.0;
    for (int i = 0; i < (int)contours.size(); i++)
    {
        double a = contourArea(contours[i]);
        if (a > bestArea)
        {
            bestArea = a;
            bestIdx = i;
        }
    }
    if (bestIdx < 0)
        return false;

    vector<Point> c = contours[bestIdx];
    double peri = arcLength(c, true);

    for (double epsFactor : {0.005, 0.01, 0.015, 0.02, 0.03})
    {
        vector<Point> approx;
        approxPolyDP(c, approx, epsFactor * peri, true);
        if (approx.size() == 4 && isContourConvex(approx))
        {
            approxOut = approx;
            return true;
        }
    }

    vector<Point> hull;
    convexHull(c, hull);
    double peri2 = arcLength(hull, true);

    for (double epsFactor : {0.005, 0.01, 0.015, 0.02, 0.03})
    {
        vector<Point> approx;
        approxPolyDP(hull, approx, epsFactor * peri2, true);
        if (approx.size() == 4 && isContourConvex(approx))
        {
            approxOut = approx;
            return true;
        }
    }
    return false;
}

// --- This function estimates table 4 corners from a segmented image; falls back to minAreaRect if needed.
static bool estimateTableCorners4(const Mat &tableSegBgr, vector<Point2f> &cornersTLTRBRBL)
{
    if (tableSegBgr.empty())
        return false;

    Mat gray;
    cvtColor(tableSegBgr, gray, COLOR_BGR2GRAY);

    Mat bin;
    threshold(gray, bin, 10, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));
    morphologyEx(bin, bin, MORPH_CLOSE, kernel);
    morphologyEx(bin, bin, MORPH_OPEN, kernel);

    vector<Point> approx;
    if (findLargestContourApprox4(bin, approx))
    {
        vector<Point2f> pts4;
        for (auto &p : approx)
            pts4.push_back(Point2f((float)p.x, (float)p.y));
        cornersTLTRBRBL = orderCornersTLTRBRBL(pts4);
        return cornersTLTRBRBL.size() == 4;
    }

    vector<vector<Point>> contours;
    findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        return false;

    int bestIdx = -1;
    double bestArea = 0.0;
    for (int i = 0; i < (int)contours.size(); i++)
    {
        double a = contourArea(contours[i]);
        if (a > bestArea)
        {
            bestArea = a;
            bestIdx = i;
        }
    }
    if (bestIdx < 0)
        return false;

    RotatedRect rr = minAreaRect(contours[bestIdx]);
    Point2f boxPts[4];
    rr.points(boxPts);

    vector<Point2f> pts4 = {boxPts[0], boxPts[1], boxPts[2], boxPts[3]};
    cornersTLTRBRBL = orderCornersTLTRBRBL(pts4);
    return cornersTLTRBRBL.size() == 4;
}

// --- This function warps the image to fixed 1224x2448 while enforcing long-side -> vertical alignment.
static bool warpToStandard(const Mat &srcBgr,
                           const vector<Point2f> &cornersTLTRBRBL,
                           Mat &warpedBgr,
                           Mat &H_out64)
{
    if (srcBgr.empty() || cornersTLTRBRBL.size() != 4)
        return false;

    const Point2f &tl = cornersTLTRBRBL[0];
    const Point2f &tr = cornersTLTRBRBL[1];
    const Point2f &br = cornersTLTRBRBL[2];
    const Point2f &bl = cornersTLTRBRBL[3];

    auto dist = [](const Point2f &a, const Point2f &b)
    {
        return (double)norm(a - b);
    };

    double srcW = 0.5 * (dist(tl, tr) + dist(bl, br));
    double srcH = 0.5 * (dist(tl, bl) + dist(tr, br));

    vector<Point2f> dstNormal = {
        Point2f(0.f, 0.f),
        Point2f((float)OUT_W - 1, 0.f),
        Point2f((float)OUT_W - 1, (float)OUT_H - 1),
        Point2f(0.f, (float)OUT_H - 1)};

    vector<Point2f> dstRot90 = {
        Point2f(0.f, 0.f),
        Point2f(0.f, (float)OUT_H - 1),
        Point2f((float)OUT_W - 1, (float)OUT_H - 1),
        Point2f((float)OUT_W - 1, 0.f)};

    const vector<Point2f> &dstPts = (srcW > srcH) ? dstRot90 : dstNormal;

    Mat H = getPerspectiveTransform(cornersTLTRBRBL, dstPts);
    H.convertTo(H_out64, CV_64F);

    warpPerspective(srcBgr, warpedBgr, H_out64, Size(OUT_W, OUT_H), INTER_LINEAR, BORDER_CONSTANT);
    return !warpedBgr.empty();
}

// --- This function builds HSV masks (red/yellow/white) on the WARPED image.
static void buildBallMasksHSV_warp(const Mat &warpedBgr, Mat &maskRed, Mat &maskYellow, Mat &maskWhite)
{
    Mat hsv;
    cvtColor(warpedBgr, hsv, COLOR_BGR2HSV);

    Mat r1, r2;
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), r1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), r2);
    maskRed = r1 | r2;

    inRange(hsv, Scalar(20, 100, 100), Scalar(35, 255, 255), maskYellow);

    // white: low S, high V
    inRange(hsv, Scalar(0, 0, 180), Scalar(180, 60, 255), maskWhite);

    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    for (Mat *m : {&maskRed, &maskYellow, &maskWhite})
    {
        morphologyEx(*m, *m, MORPH_OPEN, k);
        morphologyEx(*m, *m, MORPH_CLOSE, k);
    }
}

// --- This function extracts best ellipse center from a mask using fitEllipse and geometric filters.
static BallDet pickBestEllipseFromMask_warp(const Mat &mask,
                                            const string &name,
                                            const Scalar &bgr,
                                            double minArea,
                                            double maxArea,
                                            double minAxis,   // min ellipse axis length
                                            double maxAxis,   // max ellipse axis length
                                            double maxAspect) // max aspect ratio (major/minor)
{
    BallDet best;
    best.name = name;
    best.bgr = bgr;

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    if (contours.empty())
        return best;

    double bestScore = -1.0;

    for (auto &c : contours)
    {
        if ((int)c.size() < 20)
            continue; // fitEllipse needs enough points

        double a = contourArea(c);
        if (a < minArea || a > maxArea)
            continue;

        RotatedRect e = fitEllipse(c);
        double major = max(e.size.width, e.size.height);
        double minor = min(e.size.width, e.size.height);
        if (minor <= 1e-6)
            continue;

        if (major < minAxis || major > maxAxis)
            continue;
        if (minor < minAxis || minor > maxAxis)
            continue;

        double aspect = major / minor;
        if (aspect > maxAspect)
            continue;

        // score: prefer bigger + less distorted (aspect closer to 1)
        double s = (a) * (1.0 / aspect);
        if (s > bestScore)
        {
            bestScore = s;
            best.center_warp = e.center;
            best.valid = true;
            best.score = s;
        }
    }

    return best;
}

// --- This function picks top-2 red ellipses from mask using fitEllipse and filters.
static pair<BallDet, BallDet> pickTwoRedEllipses_warp(const Mat &maskRed,
                                                      double minArea,
                                                      double maxArea,
                                                      double minAxis,
                                                      double maxAxis,
                                                      double maxAspect)
{
    vector<BallDet> reds;

    vector<vector<Point>> contours;
    findContours(maskRed, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    for (auto &c : contours)
    {
        if ((int)c.size() < 20)
            continue;

        double a = contourArea(c);
        if (a < minArea || a > maxArea)
            continue;

        RotatedRect e = fitEllipse(c);
        double major = max(e.size.width, e.size.height);
        double minor = min(e.size.width, e.size.height);
        if (minor <= 1e-6)
            continue;

        if (major < minAxis || major > maxAxis)
            continue;
        if (minor < minAxis || minor > maxAxis)
            continue;

        double aspect = major / minor;
        if (aspect > maxAspect)
            continue;

        BallDet r;
        r.name = "Red";
        r.bgr = Scalar(0, 0, 255);
        r.center_warp = e.center;
        r.valid = true;
        r.score = a * (1.0 / aspect);
        reds.push_back(r);
    }

    sort(reds.begin(), reds.end(), [](const BallDet &a, const BallDet &b)
         { return a.score > b.score; });

    BallDet red1, red2;
    red1.name = "Red1";
    red1.bgr = Scalar(0, 0, 255);
    red2.name = "Red2";
    red2.bgr = Scalar(0, 0, 255);

    if (reds.size() >= 1)
    {
        red1 = reds[0];
        red1.name = "Red1";
    }
    if (reds.size() >= 2)
    {
        red2 = reds[1];
        red2.name = "Red2";
    }

    return {red1, red2};
}

// --- This function detects 4 balls on the WARPED image using ellipse fitting.
static vector<BallDet> detectFourBalls_warp_ellipse(const Mat &warpedBgr)
{
    Mat maskRed, maskYellow, maskWhite;
    buildBallMasksHSV_warp(warpedBgr, maskRed, maskYellow, maskWhite);

    // Warped-space bounds (1224x2448 기준) : 필요하면 여기만 튜닝하면 됨
    const double minArea = 300.0;
    const double maxArea = 80000.0;

    // axis length bounds (pixels) : 공이 너무 작거나 큰 입력이면 조정
    const double minAxis = 15.0;
    const double maxAxis = 250.0;

    // aspect ratio tolerance (warp로 타원 늘어남 감안)
    const double maxAspect = 3.0; // 더 심하게 찌그러지면 3.0까지

    auto reds = pickTwoRedEllipses_warp(maskRed, minArea, maxArea, minAxis, maxAxis, maxAspect);

    BallDet yellow = pickBestEllipseFromMask_warp(maskYellow, "Yellow", Scalar(0, 255, 255),
                                                  minArea, maxArea, minAxis, maxAxis, maxAspect);
    BallDet white = pickBestEllipseFromMask_warp(maskWhite, "White", Scalar(255, 255, 255),
                                                 minArea, maxArea, minAxis, maxAxis, maxAspect);

    vector<BallDet> out;
    out.push_back(reds.first);
    out.push_back(reds.second);
    out.push_back(yellow);
    out.push_back(white);
    return out;
}

// --- This function builds an inpaint mask from warped HSV masks (to remove distorted balls).
static Mat buildInpaintMask_warp(const Mat &warpedBgr)
{
    Mat maskRed, maskYellow, maskWhite;
    buildBallMasksHSV_warp(warpedBgr, maskRed, maskYellow, maskWhite);

    Mat maskAll = maskRed | maskYellow | maskWhite;

    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(13, 13));
    dilate(maskAll, maskAll, k);
    morphologyEx(maskAll, maskAll, MORPH_CLOSE, k);

    if (maskAll.type() != CV_8UC1)
        maskAll.convertTo(maskAll, CV_8U);
    return maskAll;
}

// --- This function estimates a reasonable standard ball radius in warped space from inpaint mask.
static int estimateBallRadiusPxFromWarpedMask(const Mat &warpedMask)
{
    vector<vector<Point>> contours;
    findContours(warpedMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<double> radii;
    for (auto &c : contours)
    {
        double a = contourArea(c);
        if (a < 300.0)
            continue;
        double r = std::sqrt(a / CV_PI);
        if (r > 5 && r < 200)
            radii.push_back(r);
    }
    if (radii.empty())
        return 32; // fallback

    sort(radii.begin(), radii.end());
    double med = radii[radii.size() / 2];

    int rp = (int)std::round(med * 0.70); // mask가 dilate된 만큼 보정
    rp = std::max(10, std::min(80, rp));
    return rp;
}

// --- This function inpaints balls out from warped image and returns a clean table surface.
static Mat inpaintBallsOut(const Mat &warpedBgr, const Mat &inpaintMask)
{
    Mat clean;
    inpaint(warpedBgr, inpaintMask, clean, 5.0, INPAINT_TELEA);
    return clean;
}

// --- This function redraws balls as perfect circles at warped centers with standard radius.
static Mat redrawBallsAsPerfectCircles(const Mat &cleanWarpedBgr,
                                       const vector<BallDet> &balls,
                                       int radiusPx)
{
    Mat out = cleanWarpedBgr.clone();

    for (const auto &b : balls)
    {
        if (!b.valid)
            continue;

        circle(out, b.center_warp, radiusPx, b.bgr, FILLED, LINE_AA);
        circle(out, b.center_warp, radiusPx, Scalar(0, 0, 0), 2, LINE_AA);

        putText(out, b.name,
                b.center_warp + Point2f(10.f, -10.f),
                FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(255, 255, 255), 2, LINE_AA);
    }
    return out;
}

// --- This function makes a display image resized with aspect ratio preserved.
static Mat makeDisplayFit(const Mat &img, int maxW = 950, int maxH = 950)
{
    if (img.empty())
        return img;
    double sx = (double)maxW / (double)img.cols;
    double sy = (double)maxH / (double)img.rows;
    double s = std::min(sx, sy);
    s = std::max(0.05, std::min(1.0, s));
    Mat out;
    resize(img, out, Size(), s, s, INTER_AREA);
    return out;
}

int main(void)
{
    Mat seg = imread("TableAndBallsOnly.png", IMREAD_COLOR);
    if (seg.empty())
    {
        cerr << "[Error] Failed to read image: " << "\n";
        return -1;
    }

    // 1) Estimate table corners
    vector<Point2f> corners;
    if (!estimateTableCorners4(seg, corners))
    {
        cerr << "[Error] Failed to estimate table corners.\n";
        return -1;
    }

    // 2) Warp to standard space
    Mat warped, H64;
    if (!warpToStandard(seg, corners, warped, H64))
    {
        cerr << "[Error] warpToStandard failed.\n";
        return -1;
    }

    // 3) Detect balls on WARPED image using ellipse fitting
    vector<BallDet> balls = detectFourBalls_warp_ellipse(warped);

    // 4) Print centers (warped coordinates)
    cout << "=== Ball centers in warped(1224x2448) coordinates ===\n";
    for (auto &b : balls)
    {
        if (!b.valid)
            cout << b.name << ": NOT FOUND\n";
        else
            cout << b.name << ": (x=" << b.center_warp.x << ", y=" << b.center_warp.y << ")\n";
    }

    // 5) Remove distorted balls via inpaint (optional but recommended)
    Mat inpaintMask = buildInpaintMask_warp(warped);
    Mat cleanWarped = inpaintBallsOut(warped, inpaintMask);

    // 6) Estimate standard radius and redraw perfect circles
    int ballR = estimateBallRadiusPxFromWarpedMask(inpaintMask);
    cout << "[Info] Standard ball radius(px) in warped space: " << ballR << "\n";

    Mat finalOut = redrawBallsAsPerfectCircles(cleanWarped, balls, ballR);

    // 7) Save outputs
    imwrite("output_warped_1224x2448.png", warped);
    imwrite("output_clean_inpaint.png", cleanWarped);
    imwrite("output_final_redrawn.png", finalOut);

    cout << "[Saved] output_warped_1224x2448.png\n";
    cout << "[Saved] output_clean_inpaint.png\n";
    cout << "[Saved] output_final_redrawn.png\n";

    // 8) Display fit
    Mat show = makeDisplayFit(finalOut);
    imshow("Final (Fit)", show);
    waitKey(0);
    return 0;
}
