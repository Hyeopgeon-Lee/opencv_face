# 생성한 CommUtils에 정의한 함수를 사용하기 위해 사용
from util.CommUtils import *

# 인식률을 높이기 위한 전처리
def preprocessing():
    # 분석하기 위한 이미지 불러오기
    image = cv2.imread("image/my_face2.jpg", cv2.IMREAD_COLOR)

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
        correction_image, correction_center = doCorrectionImage(image, face_center, eye_centers )

        # 얼굴 상세 객체(윗머리, 귀밑머리, 입술) 찾기
        rois = doDetectObject(face_center, faces[0])

        # 보정된 눈 좌표 그리기
        cv2.circle(correction_image, tuple(correction_center[0]), 5, (0, 255, 0), 2)  # 보정 눈 좌표
        cv2.circle(correction_image, tuple(correction_center[1]), 5, (0, 255, 0), 2)  # 보정 눈 좌표

        # 보정된 얼굴 줌심 그리기
        cv2.circle(correction_image, face_center, 3, (0, 0, 255), 2)  # 얼굴 중심 좌표

        resize_img = cv2.resize(correction_image, (700, 700))

        cv2.imshow("MyFace_Edit", resize_img)

    else:
        print("눈 미검출")

    # 얼굴 검출 사각형 그리기
    #cv2.rectangle(image, faces[0], (255, 0, 0), 4)

    # 이미지 크기 사이즈 변경하기
    resize_img = cv2.resize(image, (700, 700))

    # 사이즈 변경된 이미지로 출력하기
    cv2.imshow("MyFace", resize_img)

else:
    print("얼굴 미검출")

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)