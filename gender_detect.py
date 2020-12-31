# 생성한 CommUtils에 정의한 함수를 사용하기 위해 사용
from util.CommUtils import *

# 마스크 생성시, 이미지를 수치화하기 위해 사용함
import numpy as np

# 인식률을 높이기 위한 전처리
def preprocessing():
    # 분석하기 위한 이미지 불러오기
    image = cv2.imread("image/107.jpg", cv2.IMREAD_COLOR)

    # 이미지가 존재하지 안으면, 에러 반환
    if image is None: return None, None

    # 흑백사진으로 변경
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 변환한 흑백사진으로부터 히스토그램 평활화
    gray = cv2.equalizeHist(gray)

    return image, gray


# 학습된 얼굴 정면검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 학습된 눈 검출기 사용하기
eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

# 인식률을 높이기 위한 전처리 함수 호출
image, gray = preprocessing()  # 전처리

if image is None: raise Exception("영상 파일 읽기 에러")

# 얼굴 검출 수행(정확도 높이는 방법의 아래 파라미터를 조절함)
# 얼굴 검출은 히스토그램 평황화한 이미지 사용
# scaleFactor : 1.1
# minNeighbors : 인근 유사 픽셀 발견 비율이 2번 이상
# flags : 0 => 더이상 사용하지 않는 인자값
# 분석할 이미지의 최소 크기 : 가로 100, 세로 100
faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

# 얼굴이 검출되었다면,
if faces.any():

    # 얼굴 위치 값을 가져오기
    x, y, w, h = faces[0]

    # 원본이미지로부터 얼굴영역 가져오기
    face_image = image[y:y + h, x:x + w]

    # 눈 검출 수행하기(정확도 높이는 방법의 아래 파라미터를 조절함)
    # 눈 검출은 얼굴 이미지 영역만 불러와 분석 수행
    # scaleFactor : 1.15
    # minNeighbors : 인근 유사 픽셀 발견 비율이 7번 이상
    # flags : 0 => 더이상 사용하지 않는 인자값
    # 분석할 이미지의 최소 크기 : 가로 25, 세로 20
    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

    # 눈을 찾을 수 있다면,
    if len(eyes) == 2:
        # 얼굴 가운데
        face_center = (x + w // 2, y + h // 2)

        # 양쪽 눈 가운데 위치 값 가져오기
        eye_centers  = [[x+ex+ew//2, y+ey+eh//2] for ex,ey,ew,eh in eyes]

        # 사진의 기울기 보정
       #correction_image, correction_center = doCorrectionImage(image, face_center, eye_centers )
        correction_image, correction_center = doCorrectionImage(image, face_center, eye_centers)

        # 얼굴 상세 객체(윗머리, 귀밑머리, 입술) 찾기
        # rois[0] : 윗머리 / rois[1] : 귓밑머리 / rois[2] : 입술 / rois[3] : 얼굴 전체
        rois = doDetectObject(face_center, faces[0])

        # 보정된 사진 전체를 마스크만들기
        base_mask = np.full(correction_image.shape[:2], 255, np.uint8)

        # 얼굴 전체 마스크 만들기(사람의 얼굴은 평균 약 45% 타원으로 구성됨)
        # 얼굴 영역을 연산하기 않기 위해 색상을 검정색(값 : 0)으로 설정함
        face_mask = draw_ellipse(base_mask, rois[3], 0, -1)

        # 입술 마스크 만들기(사람의 얼굴은 평균 약 45% 타원으로 구성됨)
        # 입술 영역을 연산하기 위해 색상을 흰색(값 : 255)으로 설정함
        lip_mask = draw_ellipse(np.copy(base_mask), rois[2], 255)

        # 윗머리용 얼굴전체 마스크, 귓밑머리용 얼굴전체 마스크, 입술마스크, 입술 제외용 마스크를 masks 저장
        masks = [face_mask, face_mask, lip_mask, ~lip_mask]

        # 만들어 놓은 마스크에 얼굴 상세 객체의 크기에 맞게 다시 저장함
        masks = [mask[y:y + h, x:x + w] for mask, (x, y, w, h) in zip(masks, rois)]

        # 얼굴 영역별 이미지 생성
        subs = [correction_image[y:y+h, x:x+w] for x, y, w, h in rois]

        for i, sub in enumerate(subs):
            cv2.imshow('sub'+str(i), sub)

        # calcHist 파라미터 설명
        # 첫번째 파라미터(images) : 분석할 이미지 파일
        # 두번째 파라미터(Channel) : 컬러이미지(BGR)이면, 배열 값 3개로 정의
        # 세번째 파라미터(Mask) : 분석할 영역의 형태인 mask
        # 네번째 파라미터(histSize) : 히스토그램의 hist 크기, 예 : 64면 256/64 = 4 => 픽셀 4개를 1개의 픽셀로 합쳐서 연산함
        # 다섯번째 파라미터(범위) : 컬러이미지(BGR)이면 0- 256까지 배열
        hists = [cv2.calcHist([sub], [0, 1, 2], mask, (256, 256, 256), (0, 256, 0, 256, 0, 256)) for sub, mask
                 in zip(subs, masks)]

        # 각 얼굴 영역별 히스트 값의 평균 구함
        hists = [h / np.sum(h) for h in hists]

        # 얼굴색과 입술색 비교 / HISTCMP_CORREL : 1에 가까울수록 유사함
        sim1 = cv2.compareHist(hists[3], hists[2], cv2.HISTCMP_CORREL)

        # 윗머리와 귓밑머리 비교 / HISTCMP_CORREL : 1에 가까울수록 유사함
        sim2 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)

        # 여자 남자 구별하도록 만든 공식
        # 얼굴색과 입술색이 0.2보다 크면 0.2로 정의함
        criteria = 0.2 if sim1 > 0.2 else 0.1

        # 윗머리와 귓밑머리 유사도가 얼굴색과 입술색 유사도보다 크면 여성 
        value = sim2 > criteria

        # 출력 문구
        text = "Woman" if value else "Man"

        #이미지에 표기할 문구
        cv2.putText(image, text, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    else:
        print("눈 미검출")

    # 얼굴 검출 사각형 그리기
    cv2.rectangle(image, faces[0], (255, 0, 0), 4)

    # 이미지 크기 사이즈 변경하기
    #resize_img = cv2.resize(image, (700, 700))

    # 사이즈 변경된 이미지로 출력하기
    cv2.imshow("MyFace", image)

else:
    print("얼굴 미검출")

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)
