1. 이미지 파일을 불러옴
2. n개의 이미지 파일을 읽음. 한 개의 이미지 파일은 50*50이라고 가정
3. n개의 이미지 파일을 2차원 넘파이 배열로 만듦.
4. n개의 이미지 파일을 각각 1차원?...2차원?... 2500*1의 길쭉한 넘파이 배열로 바꿈.
5. 이렇게 만든 n개의 2500*1의 넘파이 배열을 좌측에서 오른쪽으로 쫙 붙여줌.
6. 이렇게 붙이면, 한 row는 모든 이미지의 같은 위치의 픽셀의 픽셀값임.
7. 그러면 한 row의 평균값을 구하면, 그 위치의 픽셀값의 평균값임.
8. 이렇게 평균값을 모든 2500개의 픽셀 위치에 대해서 구함.
9. 이 평균값을 2500*1의 배열로 만들어줌. (자연스럽게 이렇게 나오겠지)
9-1. 이 평균값 배열을 이미지로 다시 재구성하면, 모든 얼굴들의 평균 얼굴이 나올거임. 모든 얼굴들이 섞여있는 얼굴.
10. 이 평균값의 배열을 n개의 2500*1의 넘파이 배열에서 모든 열에서 빼줌.
10-1. 이렇게 빼주는 이유는, 굳이 서로 겹치는 부분에 대해서 분석해줄 필요가 없으니까. 컴퓨팅 파워만 낭비하는거니까.
11. 이렇게 평균값을 빼준 배열에서 공분산 행렬을 구해야함.
11-1. 공분산을 구하는 방법은 공분산 행렬=AA^t임.
11-2. 그런데 이렇게 구하면 데이터가 개커짐.
11-3. 그래서 A^tA로 구함.
12. 이렇게 해서 구한 공분산 행렬에서 eigenvector를 구해야함.
12-1. 이 과정도 직접 알고리즘을 짜서 하고 싶었지만, 매우 매우 복잡하고 어려운 함수가 필요하다는 것을 알게됨. (수업에서는 그냥 넘어감)
13. 구한 eigenvector를 다 사용하는 것이 아니라, 몇 개만 쓸거임.
13-1. 왜냐하면 다 사용하지 않아도 대부분의 데이터 내용을 반영할 수 있기 때문임.
14. eigenvector를 내림차순해서 위에서 몇 개의 eigenvector만 선택.
14-1. 몇 개를 선택할지도 구하는 방식이 있는데, 알아봐야함.
15. 사용할 eigenvector에 다시 평균을 뺀 배열을 곱하면 원래 배열의 eigenvector를 구할 수 있음.
16. 그 다음 그거를 사용해서 변환행렬?을 구해야함. 알아봐야함.


import numpy as np
from PIL import Image as im

import cv2
import os

import numpy as np
np.set_printoptions(threshold=np.inf)


def load_grayscaled_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (70,80))
        if resize is not None:
            images.append(resize)
    return images


def reshape_and_concatenate(np_images_to_reshape_and_concatenate):
    a = 0
    b = np.zeros(shape=(5600,1))
    for i in np_images_to_reshape_and_concatenate:
        reshaped = np.reshape(np_images_to_reshape_and_concatenate[a], (5600,1))
        b = np.concatenate((b, reshaped), axis=1)
        a += 1
    b = np.delete(b,np.s_[0:1], axis=1)
    return b


images = load_grayscaled_images_from_folder("/Users/jinchoi725/Desktop/PCA face recognition/archive/")

original_faces_rectangle = np.array(images)

original_faces_long = reshape_and_concatenate(original_faces_rectangle)

mean_face_long = original_faces_long.sum(axis=1)/410

mean_face_long = mean_face_long.reshape(5600,1)

normalized_faces_long = original_faces_long - mean_face_long

normalized_faces_long_t = np.transpose(normalized_faces_long)

covariance_matrix_1 = np.matmul(normalized_faces_long_t, normalized_faces_long)/5599
covariance_matrix_2 = np.cov(normalized_faces_long_t)

eigen_values_pca, eigen_vectors_pca = np.linalg.eig(covariance_matrix_2)

eigenvectors_of_original_faces = np.matmul(normalized_faces_long, eigen_vectors_pca)

eigenvectors_of_original_faces_T = eigenvectors_of_original_faces.T

a = 0
for i in range(410):
    example = eigenvectors_of_original_faces[:,a]
    example = example.reshape(80,70)
    test_im_list = im.fromarray(example.astype(np.uint8), "L")
    test_im_list.save("/Users/jinchoi725/Desktop/eigenfaces/eigenface_{a}.jpg".format(a = a))
    a += 1

def load_grayscaled_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (70,80))
        images.append(resize)
    return images


input_image = load_grayscaled_images_from_folder("/Users/jinchoi725/Desktop/input/")

input_image = np.array(input_image)

input_image = input_image.reshape(1*80,70)

input_image = input_image.reshape(5600, 1)

input_image_normalized_long = input_image - mean_face_long

weight_vectors_of_original_images = np.dot(eigenvectors_of_original_faces.T, normalized_faces_long)


# a=0
# weight_vectors_of_input_image = np.zeros(shape=(1,1))
# for i in range(410):
#     weight = np.matmul(eigenvectors_of_original_faces[:, a:a+1].T, input_image_normalized_long)
#     weight_vectors_of_input_image = np.concatenate((weight_vectors_of_input_image, weight), axis=1)
#     a += 1
# weight_vectors_of_input_image = np.delete(weight_vectors_of_input_image,np.s_[0:1], axis=1)
# weight_vectors_of_input_image = weight_vectors_of_input_image.T

weight_vectors_of_input_image = np.dot(eigenvectors_of_original_faces.T, input_image_normalized_long)


verify = weight_vectors_of_original_images.T - weight_vectors_of_input_image
column_sum = verify.sum(axis=1)
print(column_sum)
print(len(column_sum))


# https://github.com/jayshah19949596/Computer-Vision-Course-Assignments/blob/master/EigenFaces/EigenFaces.ipynb
# https://www.youtube.com/watch?v=SaEmG4wcFfg&t=974s&ab_channel=FrancescoPiscaniFrancescoPiscani
# https://www.youtube.com/watch?v=lnS9oCMO9NM&ab_channel=RoseRustowiczRoseRustowicz
# https://github.com/KangDoHyeong/PCA_Basic/blob/master/jupyter_notebook/0602%20%EC%8B%A4%EC%8A%B5%20%EA%B5%90%EC%95%88.ipynb
