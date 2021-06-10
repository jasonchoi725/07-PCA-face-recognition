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


# # # import necessary libraries
# # from pathlib import Path
# # from PIL import Image
# # import os, shutil
# # from os import listdir
# #
# # # image resizing
# # from PIL import Image
import numpy as np
from PIL import Image as im
# #
# # # load and display an image with Matplotlib
# # from matplotlib import image
# # from matplotlib import pyplot
# #
# #
# # input_dir = '/Users/jinchoi725/Desktop/PCA face recognition/archive/'
# # a = os.listdir(input_dir)
# #
# # # # Mac OS created a hidden file called .DS_Store which interfered with my data processing. I had to delete this hidden file, AGAIN.
# # # for root, dirs, files in os.walk('/Users/jinchoi725/Desktop/PCA face recognition/archive'):
# # #     i = 0
# # #     for file in files:
# # #         if file.endswith('.DS_Store'):
# # #             path = os.path.join(root, file)
# # #
# # #             print("Deleting: %s" % (path))
# # #
# # #             if os.remove(path):
# # #                 print("Unable to delete!")
# # #             else:
# # #                 print("Deleted...")
# # #                 i += 1
# # # print("Files Deleted: %d" % (i))
# #
# #
# # X_image_train = []
# # for fname in a:
# #     try:
# #         im = Image.open(fname)
# #         X_image_train.append(im) # The resized image files are appended to a list object X_image_train.
# #     except:
# #         pass
# #
# # print(X_image_train)
# #
# # # Using the numpy library, I converted each image file to a 2 dimentional numpy array.
# # X_image_array=[]
# # for x in range(len(X_image_train)):
# #     X_image=np.array(X_image_train[x],dtype='uint8')
# #     X_image_array.append(X_image) # The numpy arrays are appended to a list object X_image_array.
# #
# # # Checking the size of a single numpy array
# # print(X_image_array)
#
#
# image_dir = "/Users/jinchoi725/Desktop/PCA face recognition/archive/"
# for i in image_dir:
#     f = open("/Users/jinchoi725/Desktop/PCA face recognition/archive/" + i, "rb")

import cv2
import os


def load_grayscaled_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (70,70))
        if resize is not None:
            images.append(resize)
    return images


images = load_grayscaled_images_from_folder("/Users/jinchoi725/Desktop/PCA face recognition/archive/")

nparray_images = np.array(images)
print(nparray_images[0])
print(nparray_images[0].shape)


# print(np.reshape(nparray_images[0], (5600,1)))


def reshape(nparray_images):
    a = 0
    b = np.zeros(shape=(4900,1))
    for i in nparray_images:
        reshaped = np.reshape(nparray_images[a], (4900,1))
        b = np.concatenate((b, reshaped), axis=1)
        a += 1
    b = np.delete(b,np.s_[0:1], axis=1)
    return b

original_faces_arrays = reshape(nparray_images)

# b = sum(a[0])/410
# # print("b:", b)
# print("각 이미지를 늘여놓은거의 모음 리스트의 요소 수 ---->", len(a))
# b = np.array(a) # 이게 길쭉하게 5600x1로 늘어놓은 410개의 이미지 벡터를 붙여놓은 거임.
# print(b)
# print("늘여놓은 것을 넘파이 배열로 변환한 것(b)의 shape ---->", b.shape)
# print("넘파이 배열 b의 첫번째 요소 (첫번쨰 사진) ---->", b[0,0,0])

print("original_faces_arrays: ", original_faces_arrays)
print("original faces arrays.shape: ", original_faces_arrays.shape)

mean_face_array = original_faces_arrays.sum(axis=1)/410
print("mean_face_array: ", mean_face_array)
print("mean_face_array.shape: ", mean_face_array.shape)
mean_face_array = mean_face_array.reshape((4900,1))
print("mean face array.shape: ", mean_face_array.shape)
print("mean face array: ", mean_face_array)

minus_mean = original_faces_arrays - mean_face_array
print("minus mean array: ", minus_mean)
print("original minus mean array.shape: ", minus_mean.shape)

minus_mean_t = np.transpose(minus_mean)
print(minus_mean_t)
print("transpose array: ", minus_mean_t)
print("transpose array.shape: ", minus_mean_t.shape)


covariance_matrix_1 = np.matmul(minus_mean_t, minus_mean)/4899
covariance_matrix_2 = np.cov(minus_mean.T)

print("11111 \n", covariance_matrix_1)
print(covariance_matrix_1.shape)
print("22222 \n", covariance_matrix_2)
print(covariance_matrix_2.shape)

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix_2)

# print(type(eigen_values))
# print(eigen_values.shape)
# eigen_values_in_order = (-np.sort(-eigen_values))
# print(eigen_vectors)
# eigen_vectors_in_order = (-np.sort(-eigen_vectors))
# print("sort:", eigen_vectors_in_order)

# index = np.argsort(eigen_values.real)[::-1]
# eigen_values_in_order = eigen_values.real[index]
# eigen_vectors_in_order = eigen_vectors.real[:, index]

print("eigen_vectors: ", eigen_vectors)
print("eigen_vectors.shape: ", eigen_vectors.shape)
# print("eigen_vectors_in_order: ", eigen_vectors_in_order)
print("eigen_values: ", eigen_values)
# print("eigen_values_in_order: ",eigen_values_in_order)



# STEP6: Convert lower dimensionality K Eigen Vectors to Original Dimensionality
eigen_faces = np.matmul(eigen_vectors, minus_mean_t)
print(eigen_faces.shape)

eigen_faces_t = eigen_faces.T
print(eigen_faces_t)

a = 0
for i in range(410):
    example = eigen_faces_t[:,a]
    example = example.reshape(70,70)
    test_im_list = im.fromarray(example.astype(np.uint8), "L")
    test_im_list.save("/Users/jinchoi725/Desktop/eigenfaces/eigenface_{a}.jpg".format(a = a))
    a += 1

weights = np.transpose(minus_mean).dot(np.transpose(eigen_faces))
print(weights)

# test_im_list = im.fromarray(example.astype(np.uint8), "L")
# test_im_list.save("/Users/jinchoi725/Desktop/example.jpg")

# test_im_list = im.fromarray(minus_mean.astype(np.uint8), "L")
# test_im_list.save("/Users/jinchoi725/Desktop/minus_mean.jpg")


# mean_face_array = mean_face_array/410
# print(mean_face_array)
# # print(mean_face_array.shape)
# mean_face_square = mean_face_array.reshape((70,70))
# print("mean_face_square: ", mean_face_square)
# print("mean_face_square.shape: ", mean_face_square.shape)
# # #
# test_im_list = im.fromarray(mean_face_square.astype(np.uint8), "L")
# test_im_list.save("/Users/jinchoi725/Desktop/mean face square.jpg")
#
#
#
# def average(input_array):
#     b = 0
#     e = []
#     for i in range(4900):
#         a = 0
#         c = []
#         for i in input_array:
#             value = input_array[a, b, 0]
#             c.append(value)
#             a += 1
#         d = int(sum(c)/len(input_array))
#         e.append(d)
#         b += 1
#     return np.array(e)
#
# print("평균값 배열 ----> \n", average(b))
# print(average(b).shape)
#
# mean_vector = average(b).reshape((4900,1))
# print("평균값 배열 (4900,1) ----> \n", mean_vector)
# print(mean_vector.shape)
#
# mean_face = mean_vector.reshape(70,70)
# print("평균값 배열 (70,70) ----> \n", mean_face)
# print(mean_face.shape)
#
# mean_face_list = list(mean_face)
# print("mean face list:", mean_face_list)
#
# test_im_list = im.fromarray(mean_face_list[0], "L")
# test_im_list.save("/Users/jinchoi725/Desktop/test_im_list.jpg")
#
#
#
#
#
#


















