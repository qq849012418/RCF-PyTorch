# 1. 医学图像mask和原图读取

import os
import cv2
import numpy as np

import SimpleITK as sitk


def get_files_with_same_prefix(folder1, folder2):
    # 获取文件夹中的所有文件路径
    files1 = [os.path.join(folder1, file) for file in os.listdir(folder1) if
              os.path.isfile(os.path.join(folder1, file))]
    files2 = [os.path.join(folder2, file) for file in os.listdir(folder2) if
              os.path.isfile(os.path.join(folder2, file))]

    # 提取文件名前缀
    # prefix_set1 = set([os.path.basename(file).split('.')[0] for file in files1])
    prefix_set2 = set([os.path.basename(file).split('.')[0] for file in files2])

    # 找到具有相同文件名前缀的文件路径
    # common_prefixes = prefix_set1.intersection(prefix_set2)
    common_prefixes = prefix_set2

    # 存储相同文件名前缀的文件路径的数据结构，这里使用列表
    result = []

    # 遍历两个文件夹中的文件路径，将具有相同文件名前缀的文件路径存储到result中
    for prefix in common_prefixes:
        prefix_files1 = [file for file in files1 if os.path.basename(file).startswith(prefix)]
        prefix_files2 = [file for file in files2 if os.path.basename(file).startswith(prefix)]
        result.append((prefix, prefix_files1, prefix_files2))

    return result

def is_numpy_all_zeros(array):
    # 检查数组是否全为零
    return np.allclose(array, np.zeros_like(array))

def create_lst_file(file_path, content_list):
    with open(file_path, "w") as file:
        for content in content_list:
            file.write(content + "\n")

if __name__ == "__main__":

    folder1='data/Dataset502_Study45-9/imagesTr'
    folder2 = 'data/Dataset502_Study45-9/labelsTr'
    dst_forder = 'data/RCF_Study45-9'
    scale = [0.5, 1, 1.5]
    lst_file_path = os.path.join(dst_forder,"train.lst") # 新建.lst文件的路径
    content_list = []  # 要写入的内容列表

    result = get_files_with_same_prefix(folder1, folder2)

    for prefix, files1, files2 in result:
        print(f"Prefix: {prefix}")
        print(f"Files in folder1: {files1}")
        print(f"Files in folder2: {files2}")
        # 读取.nii.gz文件
        image = sitk.ReadImage(files1)
        msk = sitk.ReadImage(files2)
        # 获取图像的大小和像素间距
        size = image.GetSize()
        spacing = image.GetSpacing()
        # 将图像数据转换为numpy数组
        img_array = sitk.GetArrayFromImage(image)
        msk_array = sitk.GetArrayFromImage(msk)
        for z in range(size[2]):
            # 判断msk是否存在
            slice_msk_array = msk_array[0,z, ...]
            if is_numpy_all_zeros(slice_msk_array):
                continue
            else:
                # 提取当前横截面的图像数据
                slice_array = img_array[0,z, ...]
                # 将图像数据从浮点型转换为整型，并扩展亮度范围以在OpenCV中正确显示
                slice_array = cv2.convertScaleAbs(slice_array, alpha=(255.0 / slice_array.max()))
                slice_msk_array = cv2.convertScaleAbs(slice_msk_array, alpha=(255.0 / slice_msk_array.max()))
                # 将NumPy数组转换为OpenCV格式
                img_cv = cv2.cvtColor(slice_array, cv2.COLOR_GRAY2RGB)
                # msk_cv = cv2.cvtColor(slice_msk_array, cv2.COLOR_GRAY2RGB)
                msk_cv=slice_msk_array
                # 进行Canny边缘检测
                msk_cv = cv2.Canny(msk_cv, 100, 200)
                for k in range(len(scale)):
                    img_temp = cv2.resize(img_cv, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                    msk_temp = cv2.resize(msk_cv, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                    img_png_path=os.path.join(dst_forder,"train","img",prefix+"-"+str(z)+"-"+str(k)+".png")
                    msk_png_path=os.path.join(dst_forder, "train","gt", prefix + "-" + str(z) + "-" + str(k) + ".png")
                    cv2.imwrite(img_png_path, img_temp)
                    cv2.imwrite(msk_png_path, msk_temp)

                    content_list.append(img_png_path+" "+msk_png_path)

    create_lst_file(lst_file_path, content_list)


