import argparse
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import data_generator as dg
from models.networks import *
from toolbox import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DCEST')
    parser.add_argument('--model', default='DECENT', type=str, help='choose a type of model')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--sigma_low', default=1, type=int, help='noise sigma')
    parser.add_argument('--sigma_high', default=5, type=int, help='noise sigma')
    parser.add_argument('--zspecs', default='TrainingData\Zspectrums', type=str,
                        help='path of Z Spectrums')
    parser.add_argument('--PD_img', default='TrainingData\PD_regions', type=str, help='path of PD images')

    parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
    parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate for Adam')

    # ----------------------------------------------------------------------------------------- #
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    random.seed(2022)
    savename = args.model
    save_dir = os.path.join('train_models', savename)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    train_result = save_dir + '/' + savename + '.txt'
    with open(str(train_result), 'a') as f:
        f.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '\n')

    # ----------------------------------------------------------------------------------------- #
    batch_size = args.batch_size
    sigma1 = args.sigma_low
    sigma2 = args.sigma_high
    cuda = torch.cuda.is_available()
    n_epoch = args.epoch
    zspecs_path = os.path.join(os.path.abspath('..'), args.zspecs)
    pdimgs_path = os.path.join(os.path.abspath('..'), args.PD_img)

    seq_dirs = []
    zspecs_seqpaths = os.listdir(zspecs_path)
    for seq in zspecs_seqpaths:
        seq_dirs.append(os.path.join(zspecs_path, seq))
    pdimg_dirs = []
    pdimg_datasetpaths = os.listdir(pdimgs_path)
    for dataset in pdimg_datasetpaths:
        dataset_dirs = os.path.join(pdimgs_path, dataset)
        pdimg_slicepaths = os.listdir(dataset_dirs)
        for slice in pdimg_slicepaths:
            slice_dirs = os.path.join(dataset_dirs, slice)
            pdimg_dirs.append(slice_dirs)
    random.shuffle(pdimg_dirs)
    train_set_dirs = pdimg_dirs[499:-1]
    valid_set_dirs = pdimg_dirs[0:500]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print('===> Building model')

    if args.model == 'Local_path':
        model = Local_path()
    elif args.model == 'Global_path':
        model = Global_path()
    elif args.model == 'DECENT':
        model = DECENT()
        ## Not annotated at the first ten epochs,annotated at the last forty epochs
        # weights = OrderedDict()
        # module_lst = [i for i in model.depthconv.state_dict()]
        # ckpt = torch.load(r'F:\DCEST\train_models\sample_xiaorong\LocalPath_masked_m0/model_150.pth')['model_dict']
        # for idx, (k, v) in enumerate(ckpt.items()):
        #     if model.depthconv.state_dict()[module_lst[idx]].numel() == v.numel():
        #         weights['depthconv.'+module_lst[idx]] = v
        # module_lst = [i for i in model.spaceconv.state_dict()]
        # ckpt = torch.load(r'F:\DCEST\train_models\sample_xiaorong\UNetRes_masked_m0/model_146.pth')['model_dict']
        # for idx, (k, v) in enumerate(ckpt.items()):
        #     if model.spaceconv.state_dict()[module_lst[idx]].numel() == v.numel():
        #         weights['spaceconv.'+module_lst[idx]] = v
        # print(model.load_state_dict(weights, strict=False))
        # for name, para in model.named_parameters():
        #     if ("fcn" not in name) and ("fcn" not in name):
        #         para.requires_grad_(False)
        #     else:
        #         print("training {}".format(name))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = sum_squared_error()
    if cuda:
        model = model.cuda()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        checkpoint = torch.load((os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model.load_state_dict(checkpoint['model_dict'])
        # optimizer.load_state_dict((checkpoint['optimizer_dict']))
    print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))

    for epoch in range(initial_epoch, n_epoch):
        # train
        model.train()
        DDataset = dg.CESTnoiseDataset_sample(train_set_dirs, seq_dirs, sigma1, sigma2, m0_tag=1, mask_tag=1)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True,
                             pin_memory=True)
        train_epoch_loss = 0
        valid_epoch_loss = 0
        start_time = time.time()
        temp_time = 0
        temp_loss = 0

        for n_count, batch_yx in enumerate(DLoader):
            if cuda:
                noisy_batch, clean_batch, offset = batch_yx[0].cuda(), batch_yx[1].cuda(), batch_yx[2].cuda()
            else:
                noisy_batch, clean_batch, offset = batch_yx[0], batch_yx[1], batch_yx[2]

            if n_count == 0:
                temp_time = time.time()

            noise_img = noisy_batch
            out_train = model(noise_img)
            clean_img = clean_batch
            loss = criterion(out_train, clean_img) / 32
            train_epoch_loss += loss.item()
            temp_loss = temp_loss + loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if n_count % 20 == 19:
                cost_time = time.time() - temp_time
                temp_time = time.time()
                print('Train_EPOCH %2d CESTimgs: %4d / %4d      loss = %2.4f       cost %2dh %2dmin %2.2fs' % (
                    epoch + 1, n_count + 1, train_set_dirs.__len__() // batch_size, temp_loss / batch_size / 20,
                    cost_time // 3600,
                    cost_time % 3600 // 60, cost_time % 60))
                temp_loss = 0
        elapsed_time = time.time() - start_time
        log('train  EPOCH = %2d , loss = %5.4f , time = %2dh %2dmin %2.2fs' % (
            epoch + 1, train_epoch_loss / train_set_dirs.__len__(), elapsed_time // 3600, elapsed_time % 3600 // 60,
            elapsed_time % 60))
        time_record = str(elapsed_time // 3600) + 'h' + str(elapsed_time % 3600 // 60) + 'm' + str(
            elapsed_time % 60) + 's'
        with open(str(train_result), 'a') as f:
            f.write(
                str(epoch + 1, ) + '          train----' + str(
                    train_epoch_loss / train_set_dirs.__len__()) + '              ')

        # validation
        model.eval()
        if cuda:
            model = model.cuda()

        DDataset = dg.CESTnoiseDataset_sample(valid_set_dirs, seq_dirs, sigma1, sigma2, m0_tag=1, mask_tag=1)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=False)
        start_time = time.time()
        with torch.no_grad():
            for n_count, batch_yx in enumerate(DLoader):
                if cuda:
                    noisy_batch, clean_batch, offset = batch_yx[0].cuda(), batch_yx[1].cuda(), batch_yx[2].cuda()
                else:
                    noisy_batch, clean_batch, offset = batch_yx[0], batch_yx[1], batch_yx[2]

                if n_count == 0:
                    temp_time = time.time()

                noise_img = noisy_batch
                out_train = model(noise_img)
                clean_img = clean_batch
                loss = criterion(out_train, clean_img) / 32
                valid_epoch_loss += loss.item()

                if n_count % 10 == 9:
                    cost_time = time.time() - temp_time
                    temp_time = time.time()
                    print('Valid_EPOCH %2d CESTimgs: %4d / %4d      loss = %2.4f       cost %2dh %2dmin %2.2fs' % (
                        epoch + 1, n_count + 1, valid_set_dirs.__len__() // batch_size, loss.item() / batch_size,
                        cost_time // 3600, cost_time % 3600 // 60, cost_time % 60))

            elapsed_time = time.time() - start_time
            log('validation EPOCH = %2d , loss = %5.4f , time = %2dh %2dmin %2.2fs' % (
                epoch + 1, valid_epoch_loss / valid_set_dirs.__len__(), elapsed_time // 3600, elapsed_time % 3600 // 60,
                elapsed_time % 60))
            time_record = str(elapsed_time // 3600) + 'h' + str(elapsed_time % 3600 // 60) + 'm' + str(
                elapsed_time % 60) + 's'

            with open(str(train_result), 'a') as f:
                f.write('valid----' + str(valid_epoch_loss / valid_set_dirs.__len__()) + '\n')

        if epoch % 10 == 9:
            state = {'optimizer_dict': optimizer.state_dict(), 'model_dict': model.state_dict()}
            torch.save(state, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
