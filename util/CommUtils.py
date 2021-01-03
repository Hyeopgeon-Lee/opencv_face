import numpy as np, cv2

# 이미지 기울기 보정을 위한 함수
# 인자값1 : 원본이미지 / 인자값2 : 얼굴 중심 좌표 / 인자값3 : 양쪽 눈 중심좌표
# 결과값 : 보정된 이미지, 보정된 얼굴 중심 좌표
def doCorrectionImage(image, face_center, eye_centers):

    #양쪽 눈 좌표
    pt0, pt1 = eye_centers

    if pt0[0] > pt1[0]: pt0, pt1 = pt1, pt0

    # 두 좌표간 차분 계산
    dx, dy = np.subtract(pt1, pt0).astype(float)

    #역탄젠트로 기울기 계산
    angle = cv2.fastAtan2(dy, dx)
    
    # 계산된 기울기만큼 이미지 회전하기
    rot = cv2.getRotationMatrix2D(face_center, angle, 1)

    # 회전된 이미지를 원래 이미지 크기로 자르기
    size = image.shape[1::-1]

    # 보정된 이미지 생성
    correction_image = cv2.warpAffine(image, rot, size, cv2.INTER_CUBIC)

    # 눈 위치 보정
    eye_centers = np.expand_dims(eye_centers, axis=0)
    correction_centers = cv2.transform(eye_centers, rot)
    correction_centers = np.squeeze(correction_centers, axis=0)

    return correction_image, correction_centers

# 얼굴이미지로부터 상세 객체 탐지를 위한 함수
# 인자값1 : 얼굴 이미지 / 인자값2 : 얼굴 중심 좌표
# 결과값 : 윗머리, 귓밑머리, 입술, 얼굴 전체
def doDetectObject(face, center):
    w, h = face[2:4]
    center = np.array(center)

    # 얼굴영역 평균 비율
    # 얼굴 시작좌표 : 얼굴 증심으로부터 머리 시작좌표까지 평균 약 65%
    # 머리 끝 좌표 : 얼굴 증심으로부터 머리 끝좌표까지 평균 약 45%
    face_avg_rate = np.multiply((w,h), (0.45, 0.65))

    # 입술영역 평균 비율
    # 입술 시작좌표 : 입술 중심으로부터 얼굴높이의 평균 약 10%
    # 입술 끝 좌표 : 입술 증심으로부터 얼굴 너비의 평균 약 18%
    lib_avg_rate = np.multiply((w,h), (0.18, 0.1))

    # 얼굴 중심에서 머리 시작좌표로 이동
    pt1 = center - face_avg_rate

    # 얼굴 중심에서 머리 종료좌표로 이동
    pt2 = center + face_avg_rate

    # 얼굴 전체 영역
    face_all = roi(pt1, pt2-pt1)

    size = np.multiply(face_all[2:4], (1, 0.35))

    # 윗머리 영역
    face_up = roi(pt1, size)

    # 귀밑머리 영역
    face_down = roi(pt2-size, size)

    # 입술 중심 좌표(얼굴 중심의 약 30% 아래 위치함)
    lip_center = center + (0, h * 0.3)

    # 입술 중심에서 입술 시작좌표로 이동
    lip1 = lip_center - lib_avg_rate

    # 입술 중심에서 입술 끝좌표로 이동
    lip2 = lip_center + lib_avg_rate

    # 입술 영역
    lip = roi(lip1, lip2-lip1)

    return [face_up, face_down, lip, face_all]

# 이미지의 관심영역(ROI, Region of Interest)을 가져오기는 함수
# 인자값1 : 위치 / 인자값2 : 크기
# 결과값 : 영역 값
def roi(pt, size):
    return np.ravel([pt, size]).astype(int)

# 타원형으로 마스크 생성하는 함수
# 인자값1 : 원본 이미지 / 인자값2 : 세부 영역(윗머리, 귓밑머리, 입술, 얼굴 전체 중 하나)
# 인자값3 : 색상 / 인자값4 : 실선(사전 정의함)
# 결과값 : 타원형 마스크를 그린 이미지
def draw_ellipse(image, roi, color, thickness=cv2.FILLED):

    # 영역의 좌표 구하기
    x, y, w, h = roi

    # 영역의 자표로 부터 중심 좌표 구하기
    center = (x + w // 2, y + h // 2)

    # 타원 크기 설정하기(사람 얼굴 객체의 일반적인 타원 비율은 45%임)
    size = (int(w * 0.45), int(h * 0.45))

    # 타원 그리기
    cv2.ellipse(image, center, size, 0, 0, 360, color, thickness)

    return image


