import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import os.path as osp
import cv2
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import BSDS_Dataset
from models import RCF
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'E:\Windows\Fonts\times.ttf')
import SimpleITK as sitk
import scipy.io as sio

import time

# 用于单独保存子图的函数 https://blog.csdn.net/qq_39645262/article/details/127190982
def save_subfig(fig,ax,save_path,fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_path,fig_name), bbox_inches=extent)

def medical_image_test_multi(model, test_img, target_slice, save_dir,new_name):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    # scale = [1, 2]
    start_time = time.time()  # 程序开始时间

    # 读取.nii.gz文件
    image = sitk.ReadImage(test_img)

    # 获取图像的大小和像素间距
    size = image.GetSize()
    spacing = image.GetSpacing()
    # 将图像数据转换为numpy数组
    array = sitk.GetArrayFromImage(image)
    mat2save = np.zeros((size[0], size[1], size[2]), dtype=np.float32)
    # array = array.transpose(1,2,0)
    for z in range(size[2]):
        # 提取当前横截面的图像数据
        slice_array = array[z,...]

        # 将图像数据从浮点型转换为整型，并扩展亮度范围以在OpenCV中正确显示
        slice_array = cv2.convertScaleAbs(slice_array, alpha=(255.0/slice_array.max()))
        # 将NumPy数组转换为OpenCV格式
        img_cv = cv2.cvtColor(slice_array, cv2.COLOR_GRAY2RGB)
        # # 提高对比度的参数
        # alpha = 1.5  # 对比度增益
        # beta = 0  # 亮度增益
        # # 对图像应用对比度和亮度调整
        # img_cv = cv2.convertScaleAbs(img_cv0, alpha=alpha, beta=beta)
        # # 应用直方图均衡化
        # # 将图像拆分为三个通道
        # b, g, r = cv2.split(img_cv0)
        #
        # # 对每个通道应用直方图均衡化
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # equalized_b = clahe.apply(b)
        # equalized_g = clahe.apply(g)
        # equalized_r = clahe.apply(r)
        #
        # # 合并通道
        # img_cv = cv2.merge((equalized_b, equalized_g, equalized_r))


        slice_array = torch.from_numpy(img_cv).cuda()
        H, W, C = slice_array.shape
        r, g, b = cv2.split(img_cv)
        # 计算每个通道的均值
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)
        # mean = np.array([mean_r, mean_g, mean_b], dtype=np.float32)
        mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32) #from RCF BSDS_Dataset
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im0_ = cv2.resize(img_cv, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im1_ = im0_ - mean
            im_ = im1_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).to(torch.float32).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
        ms_fuse = ms_fuse / len(scale)

        mat2save[:,:,z]=ms_fuse

        ms_fuse = (ms_fuse * 255).astype(np.uint8)

        # 显示当前横截面
        # cv2.imshow("Slice {} ori".format(z), img_cv)
        # cv2.waitKey(0)  # 按任意键停止显示当前横截面
        # cv2.imshow("Slice {} ori2".format(z), im0_)
        # cv2.waitKey(0)  # 按任意键停止显示当前横截面
        # cv2.imshow("Slice {} preprocess".format(z), im1_)
        # cv2.waitKey(0)  # 按任意键停止显示当前横截面
        # cv2.imshow("Slice {} final".format(z), ms_fuse)
        # cv2.waitKey(0)  # 按任意键停止显示当前横截面

    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    # new_filename = test_img.replace('.nii.gz', '_cnnedge.mat')

    end_time = time.time()  # 程序结束时间
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    print('infer time:')
    print(run_time)
    new_filename = test_img.replace('.nii.gz', '_cnnedge.mat')

    # 论文作图阶段
    fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    # 定义横纵坐标的刻度
    # ax.set_yticks(range(len(yLabel)))
    # ax.set_yticklabels(yLabel, fontproperties=font)
    # ax.set_xticks(range(len(xLabel)))
    # ax.set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择hot
    slice = target_slice
    plt.imshow(mat2save[:,:,slice], cmap=plt.cm.gray)
    # # 增加右侧的颜色刻度条
    # plt.colorbar(im)
    # # 增加标题
    # plt.title('probablity map of  slice: ' + str(slice), fontproperties=font)
    # show
    plt.show()
    plt.tight_layout()
    plt.axis('off')
    plt.xticks([])

    plt.yticks([])
    save_subfig(fig, ax, save_dir, new_name+ '.png')
    # sio.savemat(
    #     new_filename,
    #     {"CNNEdgeMap":mat2save})
    print('Running medical test done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    # parser.add_argument('--checkpoint', default='bsds500_pascal_model.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--checkpoint', default='results/RCF/PUMCH2401/checkpoint_epoch10.pth', type=str, help='path to latest checkpoint')
    # parser.add_argument('--checkpoint', default='results/RCF/Study_45-9/checkpoint_epoch10.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--save-dir', help='output folder', default='results/RCF')
    parser.add_argument('--dataset', help='root folder of dataset', default='data/HED-BSDS')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)
  
    # test_dataset  = BSDS_Dataset(root=args.dataset, split='test')
    # test_loader   = DataLoader(test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False)
    # test_list = [osp.split(i.rstrip())[1] for i in test_dataset.file_list]
    # assert len(test_list) == len(test_loader)

    model = RCF().cuda()

    if osp.isfile(args.checkpoint):
        print("=> loading checkpoint from '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])


        print("=> checkpoint loaded")
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

    print('Performing the testing...')
    # single_scale_test(model, test_loader, test_list, args.save_dir)
    # multi_scale_test(model, test_loader, test_list, args.save_dir)
    # medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/PUMCH_006/PUMCH_006.nii.gz', args.save_dir)
    # medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/Study_12/Study_12.nii.gz', 19, args.save_dir,'D1P12S20')
    # medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/Study_42/Study_42.nii.gz', 34, args.save_dir,'D1P42S34')
    # medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/Study_06/Study_06.nii.gz', 39, args.save_dir,'D1P06S39')
    # medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/PUMCH_006/PUMCH_006.nii.gz', 10, args.save_dir,'D2P06S11')
    medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/PUMCH_018/PUMCH_018.nii.gz', 11, args.save_dir,'D2P18S11')
    # medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/PUMCH_024/PUMCH_024.nii.gz', 16, args.save_dir,'D2P24S17')

    # for patient_id in range(5,36,5):
    #     patient_name=str(patient_id).zfill(2)
    #     medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/OPLL_FGPM/Dataset/SHUQIANBAO_/SHUQIANBAO_'+patient_name+'_ori.nii.gz', args.save_dir)

    # for patient_id in range(6,36,6):
    #     patient_name=str(patient_id).zfill(3)
    #     medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/PUMCH_'+patient_name+'/PUMCH_'+patient_name+'.nii.gz', args.save_dir)