# 1. 전체 파이프라인 요약

본 코드는 “테이블+공만 남은 segmentation 이미지”를 입력으로 받아, **당구대를 표준 좌표계(1224×2448)로 평면 보정**한 뒤, 보정된 영상에서 **공 4개(빨강 2, 노랑 1, 흰 1)**를 검출하고, 공의 중심 좌표를 추출하여 출력한다. 또한 평면 보정 과정에서 공이 타원 형태로 왜곡되는 문제를 해결하기 위해, 공 영역을 제거(inpaint)한 뒤 **규격 원으로 재렌더링**하여 최종 결과 영상을 생성한다.

---

# 2. 단계별 파이프라인 및 주요 함수

## (A) 입력 로드

* **입력:** `TableAndBallsOnly.png` (segmentation 결과: 테이블+공만 존재)
* **코드:** `imread(..., IMREAD_COLOR)`

이 단계는 파이프라인의 시작이며, 이후 모든 처리는 “테이블 영역이 분명히 남아있다”는 전제를 기반으로 진행된다.

---

## (B) 테이블 외곽 4점 추정 (Corner Detection)

### 목적

평면 보정을 위해서는 테이블 평면의 4개의 꼭짓점(코너)이 필요하다. 이 4점이 Homography 계산의 입력이 되며, 정밀도가 전체 시스템 성능을 결정한다.

### 주요 함수 및 흐름

1. `estimateTableCorners4()`

* 입력 영상 → grayscale 변환
* 임계값(threshold)로 이진화 → 테이블 영역 마스크 생성
* 모폴로지(MORPH_CLOSE/OPEN)로 잡음 제거 및 테이블 외곽 안정화
* 외곽 Contour 기반으로 4점을 얻는 시도 → 실패 시 fallback 수행

2. `findLargestContourApprox4()`

* `findContours()`로 외곽선 검출
* 면적이 가장 큰 contour(=테이블) 선택
* `approxPolyDP()`로 다각형 근사 → **4점이 나오면 채택**
* 4점이 안 나오면 `convexHull()`을 적용 후 다시 `approxPolyDP()` 시도

3. fallback: `minAreaRect()`

* 여전히 4점 근사가 실패할 경우
* 테이블 외곽을 포함하는 최소 회전 사각형(min area rectangle)을 구하고
* 그 사각형의 4점을 코너로 사용

4. `orderCornersTLTRBRBL()`

* 구한 4점을 **항상 (tl, tr, br, bl)** 순서로 정렬한다.
* 정렬 방식:

  * (x+y) 합이 최소 → tl
  * (x+y) 합이 최대 → br
  * (x−y) 차가 최소 → tr
  * (x−y) 차가 최대 → bl

### 이 단계의 특징

* segmentation 영상이라 배경이 단순하여 contour 기반 코너 추정이 안정적이다.
* 4점 근사가 불안정할 때를 대비해 `convexHull`, `minAreaRect`로 실패 확률을 줄였다.

---

## (C) 평면 보정(Homography) 및 표준 좌표계 정규화

### 목적

입력 영상은 카메라 원근 때문에 테이블이 사다리꼴로 보일 수 있다. 이를 “정면에서 내려다본 것처럼” 펴서, 모든 프레임을 **고정 크기/고정 좌표계(1224×2448)**로 변환한다. 이후 공의 위치는 이 표준 좌표계에서 직접 비교/분석 가능하다.

### 주요 함수 및 흐름

1. `warpToStandard()`

* 입력: 원본 영상, 테이블 코너 4점
* 출력: 보정 영상 `warpedBgr`, Homography `H_out64`

2. dst 좌표(표준 테이블)

* 기본 표준 사각형:

  * (0,0), (OUT_W-1,0), (OUT_W-1, OUT_H-1), (0, OUT_H-1)

3. “긴 변/짧은 변 방향 오류” 해결 로직

* 문제 상황:
  테이블 코너를 올바르게 구했더라도, src 4점 ↔ dst 4점 대응이 잘못되면 결과가 **90도 회전**되거나 긴 변/짧은 변이 뒤바뀐 형태로 보정될 수 있다.
* 해결 방법(코드에 구현됨):

  * src에서 위/아래 변 길이 평균을 `srcW`, 좌/우 변 길이 평균을 `srcH`로 추정한다.
  * 만약 `srcW > srcH`라면(=현재 추정된 가로가 더 길게 잡히는 경우) dst 점 대응을 90도 회전한 버전(`dstRot90`)으로 바꿔 넣는다.
  * 이를 통해 **테이블의 긴 변이 항상 출력 영상에서 세로(2448) 방향으로 가도록 강제**한다.

4. Homography 계산 및 warp

* `getPerspectiveTransform(srcPts, dstPts)`로 H 계산
* `warpPerspective()`로 표준 크기(1224×2448)에 맞춰 보정

---

# 3. 평면 보정에서 발생하는 대표 오류와 해결 방식

## 오류 1) 긴 변/짧은 변이 바뀌는 90도 회전 문제

### 원인

* 4점 정렬(tl,tr,br,bl)이 되어도 “dst 사각형에 대응시키는 순서”가 고정이면,
* 테이블이 세로로 길어야 하는데 가로로 길게 보정되는 문제가 생길 수 있다.

### 해결 (코드의 핵심 로직)

* src에서 테이블의 장축/단축을 길이 기반으로 판별하고,
* dst 점 배열을 정상 버전과 90도 회전 버전 중 하나로 선택한다.

즉, “코너를 잘 잡는 것”뿐 아니라 “올바른 방향으로 매핑되는지”를 자동으로 검사하는 안전장치를 둔 것이다.

---

## 오류 2) 공이 원이 아니라 타원으로 늘어나는 문제

### 원인

* Homography는 “테이블 평면”을 펴는 변환이다.
* 공은 테이블 위에 높이를 가진 **구체(sphere)**이므로, 평면 보정 후에는 공의 영상이 등방적으로 유지되지 않고 타원 형태로 왜곡될 수 있다.
* 또한 warp 과정의 샘플링/보간(interpolation) 영향도 타원 왜곡을 강화할 수 있다.

### 해결 전략(코드에서 채택한 방식)

1. **검출 단계에서는 원 검출이 아니라 타원 검출을 수행**

* `fitEllipse()` 기반으로 타원 중심을 직접 추정한다.
* `detectFourBalls_warp_ellipse()`에서 색 마스크 → contour → fitEllipse → 중심 추출

2. **출력(시각화) 단계에서는 타원 공을 제거 후 규격 원으로 다시 그림**

* `buildInpaintMask_warp()`로 공 영역 마스크 생성(+dilate)
* `inpaintBallsOut()`로 공을 지워 테이블 천 텍스처로 복원
* `estimateBallRadiusPxFromWarpedMask()`로 표준 반지름을 자동 추정
* `redrawBallsAsPerfectCircles()`에서 중심 위치에 일정 반지름 원을 다시 렌더링

이 방식은 “기하학적으로 구체가 평면 보정에서 타원으로 보이는 것은 불가피하다”는 점을 인정하고, **중심 좌표만 정확히 확보한 뒤 렌더링을 표준화**함으로써 결과의 일관성과 가독성을 확보한다.

---

# 4. 공 검출 파이프라인(보정 영상 기준)

## (D) HSV 기반 색 마스크 생성

* `buildBallMasksHSV_warp()`
* Red: HSV 2구간 합집합
* Yellow: 단일 범위
* White: 낮은 S + 높은 V

이후 `morphologyEx(OPEN/CLOSE)`로 잡음을 제거하여 contour 검출의 안정성을 높였다.

---

## (E) 타원 기반 중심 추출 (fitEllipse)

### 주요 함수

* 노랑/흰: `pickBestEllipseFromMask_warp()`

  * contour 면적(minArea~maxArea)
  * 축 길이(minAxis~maxAxis)
  * 종횡비(aspect ratio) 제한(maxAspect)
  * score = area / aspect 로 “크고 덜 찌그러진 타원” 우선 선택

* 빨강 2개: `pickTwoRedEllipses_warp()`

  * 같은 필터로 후보를 모두 모은 뒤 score 상위 2개를 Red1/Red2로 채택

---

# 5. 출력(후처리 및 결과 생성)

## (F) 공 제거 및 복원(Inpainting)

* `buildInpaintMask_warp()`에서 마스크를 확장해 공 영역을 충분히 덮음
* `inpaintBallsOut()`에서 TELea 방식으로 복원

## (G) 규격 원으로 재렌더링

* `estimateBallRadiusPxFromWarpedMask()`

  * 마스크 contour 면적 → 등가 원 반지름 추정 → median 기반으로 안정화
* `redrawBallsAsPerfectCircles()`

  * 중심에 (반지름=ballR)로 색 동일하게 채운 원을 그리고 외곽선을 추가

## (H) 저장 및 화면 표시

* 저장:

  * `output_warped_1224x2448.png`
  * `output_clean_inpaint.png`
  * `output_final_redrawn.png`
* 화면:

  * `makeDisplayFit()`로 비율 유지 축소 후 표시

---

# 6. 결론 
본 시스템은 테이블 외곽을 contour 기반으로 4점 추정한 뒤, Homography로 표준 좌표계(1224×2448)로 정규화하여 입력 영상의 원근 변화를 제거하였다. 이때 코너 매핑 과정에서 발생할 수 있는 긴 변/짧은 변 뒤바뀜 문제를 길이 기반으로 자동 판별하여 올바른 방향으로 보정되도록 설계하였다. 또한 공은 평면 보정 후 타원으로 왜곡되므로 원 검출 대신 `fitEllipse` 기반의 타원 검출로 중심점을 안정적으로 추정하고, 최종 출력에서는 공 영역을 inpaint로 제거한 뒤 규격 원을 재렌더링함으로써 시각적 일관성과 좌표 추출 정확도를 동시에 확보하였다.

