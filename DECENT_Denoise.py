import time
import numpy as np
import scipy
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from toolbox import *
from models.networks import *
import scipy.io as io

m0_tag = True
mask_tag = True

datapath = r'Data/numerical_simulation'
# load model
model = DECENT()  # choose model
save_name = 'DECENT'  # save name
model_dict = torch.load('models_repo/DECENT.pth', map_location=torch.device('cpu'))  # load weight
model.load_state_dict(model_dict)
if torch.cuda.is_available():
    model = model.cuda()
model.eval()


# load data
noisy_datapath = datapath + '/noisy_img.mat'
noisy_m0path = datapath + '/noisy_m0.mat'
clean_datapath = datapath + '/clean_img.mat'
clean_m0path = datapath + '/m0.mat'
mask_path = datapath + '/mask.mat'
noisy_imgs = io.loadmat(noisy_datapath)['noisy_img']
noisy_m0 = io.loadmat(noisy_m0path)['noisy_m0']
clean_imgs = io.loadmat(clean_datapath)['clean_img']
clean_m0 = io.loadmat(clean_m0path)['m0']
mask = io.loadmat(mask_path)['mask']

noisy_imgs = noisy_imgs.transpose(2, 0, 1)
clean_imgs = clean_imgs.transpose(2, 0, 1)

if m0_tag:
    clean_imgs[0, :, :] = clean_m0
    noisy_imgs[0, :, :] = noisy_m0
if mask_tag:
    noisy_imgs = noisy_imgs * mask
    clean_imgs = clean_imgs * mask

# padding alone spectral dimension
temp = np.zeros([((noisy_imgs.shape[0]-1)//8 + 1)*8, noisy_imgs.shape[1], noisy_imgs.shape[2]])
temp[0:noisy_imgs.shape[0], ...] = noisy_imgs

with torch.no_grad():
    start_time = time.time()
    clean_data = torch.from_numpy(np.array(clean_imgs).astype('float32'))
    noisy_data = torch.from_numpy(np.array(temp).astype('float32'))
    noisy_data = torch.unsqueeze(noisy_data, 0)
    noisy_data = torch.unsqueeze(noisy_data, 0)
    if torch.cuda.is_available():
        noisy_data = noisy_data.cuda()
    out_train = model(noisy_data)
    noisy_data = noisy_data.cpu().detach().numpy().squeeze()
    clean_data = clean_data.detach().numpy().squeeze()
    denoi_data = out_train.cpu().detach().numpy().squeeze()
    elapsed_time = time.time() - start_time

temp = denoi_data[0:noisy_imgs.shape[0]]
denoi_data = temp

# evaluation
psnr_noisy = peak_signal_noise_ratio(clean_data, noisy_data)
psnr_denoi = peak_signal_noise_ratio(clean_data, denoi_data)
ssim_noisy = structural_similarity(clean_data, noisy_data)
ssim_denoi = structural_similarity(clean_data, denoi_data)
mse_noisy = mean_squared_error(clean_data, noisy_data)
mse_denoi = mean_squared_error(clean_data, denoi_data)
# log
print('noisy-----psnr:%2.4f  -----ssim:%.4f -----mse:%.10f' % (psnr_noisy, ssim_noisy, mse_noisy))
print('denoi-----psnr:%2.4f  -----ssim:%.4f -----mse:%.10f' % (psnr_denoi, ssim_denoi, mse_denoi))
print('loss : %.8f' % (mse_denoi * clean_m0.shape[0] * clean_m0.shape[1] / 2))
log('take time = %2dh %2dmin %2.4fs' % (elapsed_time // 3600, elapsed_time % 3600 // 60, elapsed_time % 60))
# save denoised data
denoi_data = denoi_data.transpose(1, 2, 0)
scipy.io.savemat(datapath + '/' + save_name + '.mat', {save_name: denoi_data.astype(np.float64)})

