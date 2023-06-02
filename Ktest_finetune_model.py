import os
import numpy as np
import os.path as osp
import cv2
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import BSDS_Dataset
from models import RCF

import SimpleITK as sitk
import scipy.io as sio

def single_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        all_res = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          all_res[i, 0, :, :] = results[i]
        filename = osp.splitext(test_list[idx])[0]
        torchvision.utils.save_image(1 - all_res, osp.join(save_dir, '%s.jpg' % filename))
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ss.png' % filename), fuse_res)
        #print('\rRunning single-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running single-scale test done')


def multi_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for idx, image in enumerate(test_loader):
        in_ = image[0].numpy().transpose((1, 2, 0))
        _, _, H, W = image.shape
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
        ms_fuse = ms_fuse / len(scale)
        ### rescale trick
        # ms_fuse = (ms_fuse - ms_fuse.min()) / (ms_fuse.max() - ms_fuse.min())
        filename = osp.splitext(test_list[idx])[0]
        ms_fuse = ((1 - ms_fuse) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ms.png' % filename), ms_fuse)
        #print('\rRunning multi-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running multi-scale test done')

#keenster 2023.5.8
def medical_image_test(model, test_img, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    # 读取.nii.gz文件
    image = sitk.ReadImage(test_img)
    mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
    # 获取图像的大小和像素间距
    size = image.GetSize()
    spacing = image.GetSpacing()
    # 将图像数据转换为numpy数组
    array = sitk.GetArrayFromImage(image)
    # array = array.transpose(1,2,0)
    for z in range(size[0]):
        # 提取当前横截面的图像数据
        slice_array = array[z,...]

        # 将图像数据从浮点型转换为整型，并扩展亮度范围以在OpenCV中正确显示
        slice_array = cv2.convertScaleAbs(slice_array, alpha=(255.0/slice_array.max()))
        # 将NumPy数组转换为OpenCV格式
        img_cv = cv2.cvtColor(slice_array, cv2.COLOR_GRAY2RGB)
        img_cv =(img_cv - mean)
        slice_array = torch.from_numpy(img_cv).cuda()
        H, W,C = slice_array.shape
        slice_array = slice_array.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        # slice_array=slice_array.reshape(1,C,H, W)
        results = model(slice_array)
        all_res = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            all_res[i, 0, :, :] = results[i]
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = (fuse_res * 255).astype(np.uint8)

        # 显示当前横截面
        cv2.imshow("Slice {}".format(z), fuse_res)
        cv2.waitKey(0)  # 按任意键停止显示当前横截面

    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    print('Running medical test done')

def medical_image_test_multi(model, test_img, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    # scale = [1, 2]

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
    new_filename = test_img.replace('.nii.gz', '_cnnedge.mat')
    sio.savemat(
        new_filename,
        {"CNNEdgeMap":mat2save})
    print('Running medical test done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    # parser.add_argument('--checkpoint', default='bsds500_pascal_model.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--checkpoint', default='results/RCF/checkpoint_epoch10.pth', type=str, help='path to latest checkpoint')
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
    # medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/XH_01/XH_01.nii.gz', args.save_dir)
    medical_image_test_multi(model, 'D:/Keenster/MatlabScripts/KeensterSSM/Study_54/Study_54.nii.gz', args.save_dir)
