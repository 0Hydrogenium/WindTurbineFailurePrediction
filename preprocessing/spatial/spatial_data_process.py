import copy
import math

import numpy as np
import pandas as pd
import cv2

from preprocessing import SqlMethods


methods = SqlMethods()


# 风机空间数据处理

img = cv2.imread("../../sql_data/input_map.png")

result = copy.deepcopy(img)

pixel_list = np.argwhere(img == [0, 0, 0])[:, :-1]

pixel_list = np.array(list(set([tuple(x) for x in pixel_list]))).tolist()

pixel_group_list = []
while True:
    if not pixel_list:
        break

    base_pixel = pixel_list[0]
    group_base_list = []

    for i in range(len(pixel_list)-1, -1, -1):
        if math.sqrt(math.pow(pixel_list[i][0] - base_pixel[0], 2) +
                     math.pow(pixel_list[i][1] - base_pixel[1], 2)) <= 6:
            group_base_list.append(np.array(pixel_list.pop(i)))

    pixel_group_list.append(np.array(group_base_list))

centers_list = []
for group in pixel_group_list:
    center_x = int(np.sum(group[:, 0]) / len(group[:, 0]))
    center_y = int(np.sum(group[:, 1]) / len(group[:, 1]))
    centers_list.append([center_x, center_y])

distance_array = []
for i in range(len(centers_list)):
    row_list = []
    for j in range(len(centers_list)):
        distance = math.sqrt(math.pow(centers_list[j][0] - centers_list[i][0], 2) +
                             math.pow(centers_list[j][1] - centers_list[i][1], 2))
        row_list.append(distance)
    distance_array.append(np.array(row_list))
distance_array = np.array(distance_array)

pd.DataFrame(
    data=distance_array[:, :],
    columns=list(range(1, len(distance_array)+1)),
    index=list(range(1, len(distance_array)+1))
).to_csv("../../sql_data/spatial_data.csv", encoding="gbk")

img_copy = copy.deepcopy(img)

for c in centers_list:
    cv2.circle(img=img_copy, center=(int(c[1]), int(c[0])), radius=7, color=(220, 20, 60), thickness=1)

cv2.imshow("w", img)
cv2.imshow("raw", img_copy)
cv2.waitKey(0)
