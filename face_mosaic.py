# 설치한 OpenCV 패키지 불러오기
import cv2

#인식률을 높이기 위한 전처리
def preprocessing():

    # 분석하기 위한 이미지 불러오기
    image = cv2.imread("image/my_face2.jpg", cv2.IMREAD_COLOR)
    
    # 이미지가 존재하지 안으면, 에러 반환
    if image is None: return None, None

    # 이미지 크기 사이즈 변경하기
    image = cv2.resize(image, (700, 700))

    # 흑백사진으로 변경
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 변환한 흑백사진으로부터 히스토그램 평활화
    gray = cv2.equalizeHist(gray)

    return image, gray

# 학습된 얼굴 정면검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

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

    #얼굴 위치 값을 가져오기
    x, y, w, h = faces[0]

    #원본이미지로부터 얼굴영역 가져오기
    face_image = image[y:y + h, x:x + w]

    #모자이크 비율(픽셀크기 증가로 모자이크 만들기)
    mosaic_rate = 30

    # 얼굴 영역의 픽셀을 mosaic_rate에 따라 나눠서 픽셀 확대
    face_image = cv2.resize(face_image, (w // mosaic_rate, h // mosaic_rate))

    # 확대한 얼굴 이미지(픽셀)을 얼굴 크기에 덮어쓰기
    face_image = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_AREA)

    # 원본이미지에 모자이크 처리한 얼굴 이미지 붙이기
    image[y:y + h, x:x + w] = face_image;

    # 모자이크 처리된 이미지 파일 생성하기
    cv2.imwrite("result/my_image_mosaic.jpg", image)

    # 모자이크 처리된 이미지 보여주기
    cv2.imshow("mosaic_image", cv2.imread("result/my_image_mosaic.jpg", cv2.IMREAD_COLOR))

else: print("얼굴 미검출")

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)