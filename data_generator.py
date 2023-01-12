import glob
import os
import random

import imageio
import numpy as np
from torch.utils.data import Dataset
import torch
from random import randint
import scipy.io as io


class CESTnoiseDataset_sample(Dataset):
    def __init__(self, pdimg_dirs, seq_dirs, sigmalow, sigmahigh, m0_tag=0, mask_tag=0):
        super(CESTnoiseDataset_sample, self).__init__()
        self.pdimg_dirs = pdimg_dirs
        self.seq_dirs = seq_dirs
        self.sigmalow = sigmalow
        self.sigmahigh = sigmahigh
        self.m0_tag = m0_tag
        self.mask_tag = mask_tag

    def __getitem__(self, index):
        m0_tag = self.m0_tag
        mask_tag = self.mask_tag

        ppm = np.random.choice(range(25, 33))
        width = 128

        if m0_tag == 1:
            ppm = ppm - 1
        clean_imgs = np.zeros([ppm, width, width])

        # load m0 and regions mask
        pd = glob.glob(self.pdimg_dirs[index] + '/*PD.png')  # matrix size:256*256
        wm = glob.glob(self.pdimg_dirs[index] + '/*WM.png')  # region_1
        gm = glob.glob(self.pdimg_dirs[index] + '/*GM.png')  # region_2
        csf = glob.glob(self.pdimg_dirs[index] + '/*CSF.png')  # region_3
        pd = imageio.imread(pd[0]) / 255
        wm = imageio.imread(wm[0]) / 255
        gm = imageio.imread(gm[0]) / 255
        csf = imageio.imread(csf[0]) / 255
        pd = pd[0:-1:2, 0:-1:2]
        wm = wm[0:-1:2, 0:-1:2]
        gm = gm[0:-1:2, 0:-1:2]
        csf = csf[0:-1:2, 0:-1:2]
        other = np.ones([width, width]) - wm - gm - csf  # region_4
        mask = pd > 0.1

        # load Z-spectrum
        seq = np.random.choice(range(0, 40))  # seq number
        zspecs_dir = self.seq_dirs[seq]
        zspecs = np.random.choice(range(0, 40), size=4, replace=False)
        zspecs = zspecs + 40
        zspec_dirs = os.listdir(zspecs_dir)
        zspec_dirs.sort()
        WM_specs = io.loadmat(os.path.join(zspecs_dir, zspec_dirs[zspecs[0]]))['zspecs']  # region_1
        GM_specs = io.loadmat(os.path.join(zspecs_dir, zspec_dirs[zspecs[1]]))['zspecs']  # region_2
        CSF_specs = io.loadmat(os.path.join(zspecs_dir, zspec_dirs[zspecs[2]]))['zspecs']  # region_3
        OTHER_specs = io.loadmat(os.path.join(zspecs_dir, zspec_dirs[zspecs[3]]))['zspecs']  # region_4

        # random sample from Z spectrum
        sample_tag = np.random.choice(range(0, 6))  # for global and fusion pathway
        # sample_tag = np.random.choice(range(3, 6))  # for spectral pathway
        if sample_tag == 0:
            offset = random.sample(range(0, WM_specs.shape[0] // 2 + 1), ppm)
            offset.sort()
        elif sample_tag == 1:
            offset = random.sample(range(WM_specs.shape[0] // 2, WM_specs.shape[0]), ppm)
            offset.sort()
        elif sample_tag == 2:
            offset = random.sample(range(WM_specs.shape[0]), ppm)
            offset.sort()
        else:
            offset = randint(0, WM_specs.shape[0] - ppm)
            offset = range(offset, offset + ppm)

        # generate CEST images
        pd_wm = pd * wm
        pd_gm = pd * gm
        pd_csf = pd * csf
        pd_other = pd * other

        for w in range(width):
            for h in range(width):
                if pd_wm[h, w] != 0:
                    clean_imgs[:, h, w] = pd_wm[h, w] * WM_specs[offset, randint(0, WM_specs.shape[1] - 1)]
                if pd_gm[h, w] != 0:
                    clean_imgs[:, h, w] = pd_gm[h, w] * GM_specs[offset, randint(0, GM_specs.shape[1] - 1)]
                if pd_csf[h, w] != 0:
                    clean_imgs[:, h, w] = pd_csf[h, w] * CSF_specs[offset, randint(0, CSF_specs.shape[1] - 1)]
                if pd_other[h, w] != 0:
                    clean_imgs[:, h, w] = pd_other[h, w] * OTHER_specs[offset, randint(0, OTHER_specs.shape[1] - 1)]

        temp = np.zeros([31, width, width])
        temp[0:ppm, ...] = clean_imgs
        clean_imgs = temp

        # add Rician noise
        sigma = randint(self.sigmalow, self.sigmahigh)

        clean_imgs = np.array(clean_imgs, dtype='float32')
        pd = np.expand_dims(pd, axis=0)

        if m0_tag == 1:
            clean_imgs = np.concatenate((pd, clean_imgs), axis=0)

        noise = np.random.rayleigh(sigma / 255, size=clean_imgs.shape)
        noise = noise - np.mean(noise)
        noisy_imgs = clean_imgs + noise

        if mask_tag == 1:
            noisy_imgs = noisy_imgs * mask
            clean_imgs = clean_imgs * mask

        clean_imgs = torch.from_numpy(np.array(clean_imgs).astype('float32'))
        noisy_imgs = torch.from_numpy(np.array(noisy_imgs).astype('float32'))
        clean_imgs = torch.unsqueeze(clean_imgs, 0)
        noisy_imgs = torch.unsqueeze(noisy_imgs, 0)
        return noisy_imgs, clean_imgs, ppm

    def __len__(self):
        return self.pdimg_dirs.__len__()
