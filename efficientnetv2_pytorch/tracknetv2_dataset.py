import os
import math
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class TrackNetV2Dataset(Dataset):
    def __init__(self, csv_path, root_dir='', input_height=360, input_width=640):
        """
        csv_path: 標註檔案的完整路徑
        root_dir: 影像與標註的根目錄（可為空，則路徑以csv為主）
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.height = input_height
        self.width = input_width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path, path_prev, path_preprev, path_gt, x, y, status, vis = row

        # 拼接路徑
        img_path = os.path.join(self.root_dir, str(path))
        img_prev_path = os.path.join(self.root_dir, str(path_prev))
        img_preprev_path = os.path.join(self.root_dir, str(path_preprev))
        gt_path = os.path.join(self.root_dir, str(path_gt))

        # 處理缺失值
        if pd.isna(x):
            x = -1
            y = -1

        # 讀取三幀影像並堆疊
        img = cv2.imread(img_path)
        img_prev = cv2.imread(img_prev_path)
        img_preprev = cv2.imread(img_preprev_path)
        img = cv2.resize(img, (self.width, self.height))
        img_prev = cv2.resize(img_prev, (self.width, self.height))
        img_preprev = cv2.resize(img_preprev, (self.width, self.height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)  # (9, H, W)

        # 讀取 GT heatmap
        gt = cv2.imread(gt_path)
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt[:, :, 0]
        # 修改這一行，保持 gt 為 (H, W)
        # gt = np.reshape(gt, (self.width * self.height))
        gt = gt.astype(np.float32) / 255.0  # 標準化到0~1
        gt = np.expand_dims(gt, axis=0)     # (1, H, W)
        return imgs, gt, x, y, vis