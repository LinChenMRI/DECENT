import torch
import torch.nn as nn
import models.basicblock as b


class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=None, nb=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        if nc is None:
            nc = [64, 128, 256, 512]
        self.m_head = b.conv(in_nc, nc[0], kernel_size=3, stride=1, padding=1, bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = b.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = b.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = b.downsample_strideconv
        elif downsample_mode == 'strideconv122':
            downsample_block = b.downsample_strideconv122
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = b.sequential(
            *[b.ResBlock(nc[0], nc[0], kernel_size=3, stride=1, padding=1, bias=False, mode='C' + act_mode + 'C') for _
              in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = b.sequential(
            *[b.ResBlock(nc[1], nc[1], kernel_size=3, stride=1, padding=1, bias=False, mode='C' + act_mode + 'C') for _
              in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = b.sequential(
            *[b.ResBlock(nc[2], nc[2], kernel_size=3, stride=1, padding=1, bias=False, mode='C' + act_mode + 'C') for _
              in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = b.sequential(
            *[b.ResBlock(nc[3], nc[3], kernel_size=3, stride=1, padding=1, bias=False, mode='C' + act_mode + 'C') for _
              in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = b.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = b.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = b.upsample_convtranspose
        elif upsample_mode == 'convtranspose122':
            upsample_block = b.upsample_convtranspose122
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = b.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[b.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = b.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[b.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = b.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[b.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = b.conv(nc[0], out_nc, kernel_size=3, stride=1, padding=1, bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        # x = x + x1
        return x


class UNetRes_res(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=None, nb=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(UNetRes_res, self).__init__()

        if nc is None:
            nc = [64, 128, 256, 512]
        self.m_head = b.conv(in_nc, nc[0], kernel_size=3, stride=1, padding=1, bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = b.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = b.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = b.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = b.sequential(
            *[b.ResBlock(nc[0], nc[0], kernel_size=3, stride=1, padding=1, bias=False, mode='C' + act_mode + 'C') for _
              in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = b.sequential(
            *[b.ResBlock(nc[1], nc[1], kernel_size=3, stride=1, padding=1, bias=False, mode='C' + act_mode + 'C') for _
              in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = b.sequential(
            *[b.ResBlock(nc[2], nc[2], kernel_size=3, stride=1, padding=1, bias=False, mode='C' + act_mode + 'C') for _
              in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = b.sequential(
            *[b.ResBlock(nc[3], nc[3], kernel_size=3, stride=1, padding=1, bias=False, mode='C' + act_mode + 'C') for _
              in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = b.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = b.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = b.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = b.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[b.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = b.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[b.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = b.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[b.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = b.conv(nc[0], out_nc, kernel_size=3, stride=1, padding=1, bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x0 - x

# 两条路径逐像素点直接相加
class DCEST_gl(nn.Module):
    def __init__(self):
        super(DCEST_gl, self).__init__()
        self.spaceconv = Global_path()
        self.depthconv = Local_path()

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x = x1 + x2
        return x


class DCEST_dru_gl_3FCN(nn.Module):
    def __init__(self):
        super(DCEST_dru_gl_3FCN, self).__init__()
        self.spaceconv = Global_path()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(3, 64, kernel_size=1, stride=1, padding=0, bias=False, mode='CR') ,
                                *[b.ResBlock(64, 64, kernel_size=1, stride=1, padding=0, bias=False, mode='CRC') for _ in range(3)],
                                b.conv(64, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='CR'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x0, x1, x2), 1)
        x = self.fcn(x3)
        return x


class DCEST_res_gl(nn.Module):
    def __init__(self):
        super(DCEST_res_gl, self).__init__()
        self.spaceconv = Global_path()
        self.depthconv = Local_path()

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x = x1 + x2
        return x0 - x


class DCEST_dru(nn.Module):
    def __init__(self):
        super(DCEST_dru, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x = x1 + x2
        return x


class DCEST_res_dru(nn.Module):
    def __init__(self):
        super(DCEST_res_dru, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x = x1 + x2
        return x0 - x

# 两条路径通过FCN融合
class DCEST_dru_CR(nn.Module):
    def __init__(self):
        super(DCEST_dru_CR, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(2, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='CR'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x


class DCEST_dru_CR_3FCN(nn.Module):
    def __init__(self):
        super(DCEST_dru_CR_3FCN, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(2, 64, kernel_size=1, stride=1, padding=0, bias=False, mode='CR') ,
                                *[b.ResBlock(64, 64, kernel_size=1, stride=1, padding=0, bias=False, mode='CRC') for _ in range(3)],
                                b.conv(64, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='CR'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x


class DCEST_dru_CR_3FCN_3c(nn.Module):
    def __init__(self):
        super(DCEST_dru_CR_3FCN_3c, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(3, 64, kernel_size=1, stride=1, padding=0, bias=False, mode='CR') ,
                                *[b.ResBlock(64, 64, kernel_size=1, stride=1, padding=0, bias=False, mode='CRC') for _ in range(3)],
                                b.conv(64, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='CR'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x0, x1, x2), 1)
        x = self.fcn(x3)
        return x


class DCEST_dru_CR_3FCN_NR(nn.Module):
    def __init__(self):
        super(DCEST_dru_CR_3FCN_NR, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(2, 64, kernel_size=1, stride=1, padding=0, bias=False, mode='CR') ,
                                *[b.ResBlock(64, 64, kernel_size=1, stride=1, padding=0, bias=False, mode='CRC') for _ in range(3)],
                                b.conv(64, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='C'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x


class DCEST_dru_CR_3FCN_v2(nn.Module):
    def __init__(self):
        super(DCEST_dru_CR_3FCN_v2, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, mode='CR') ,
                                *[b.ResBlock(64, 64, kernel_size=3, stride=1, padding=1, bias=False, mode='CRC') for _ in range(4)],
                                b.conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, mode='CR'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x


class DCEST_dru_res_CR(nn.Module):
    def __init__(self):
        super(DCEST_dru_res_CR, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(2, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='CR'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x0 - x


class DCEST_dru_C(nn.Module):
    def __init__(self):
        super(DCEST_dru_C, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(2, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='C'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x


class DCEST_dru_res_C(nn.Module):
    def __init__(self):
        super(DCEST_dru_res_C, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path()
        self.fcn = b.sequential(b.conv(2, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='C'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x0 - x


class DCEST_dru_122(nn.Module):
    def __init__(self):
        super(DCEST_dru_122, self).__init__()
        self.spaceconv = UNetRes(downsample_mode='strideconv122',
                 upsample_mode='convtranspose122')
        self.depthconv = Local_path(downsample_mode='strideconv122',
                 upsample_mode='convtranspose122')

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x = x1 + x2
        return x


class DCEST_dru_CR122(nn.Module):
    def __init__(self):
        super(DCEST_dru_CR122, self).__init__()
        self.spaceconv = UNetRes(in_nc=1, out_nc=1, downsample_mode='strideconv122',
                 upsample_mode='convtranspose122')
        self.depthconv = Local_path(in_nc=1, out_nc=1, downsample_mode='strideconv122',
                 upsample_mode='convtranspose122')
        self.fcn = b.sequential(b.conv(2, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='CR'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x


class DCEST_dru_CR_local211(nn.Module):
    def __init__(self):
        super(DCEST_dru_CR_local211, self).__init__()
        self.spaceconv = UNetRes()
        self.depthconv = Local_path(downsample_mode='strideconv211',
                 upsample_mode='convtranspose211')
        self.fcn = b.sequential(b.conv(2, 1, kernel_size=1, stride=1, padding=0, bias=False, mode='CR'))

    def forward(self, x0):
        x1 = self.spaceconv(x0)
        x2 = self.depthconv(x0)
        x3 = torch.cat((x1, x2), 1)
        x = self.fcn(x3)
        return x


class UNetRes_CR122(nn.Module):
    def __init__(self):
        super(UNetRes_CR122, self).__init__()
        self.spaceconv = UNetRes(downsample_mode='strideconv122', upsample_mode='convtranspose122')

    def forward(self, x0):
        x = self.spaceconv(x0)
        return x
# ----------------------------------#


class Global_path(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=None, nb=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(Global_path, self).__init__()

        if nc is None:
            nc = [64, 128, 256, 512]
        self.m_head = b.conv(in_nc, nc[0], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = b.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = b.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = b.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = b.sequential(*[
            b.ResBlock(nc[0], nc[0], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = b.sequential(*[
            b.ResBlock(nc[1], nc[1], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = b.sequential(*[
            b.ResBlock(nc[2], nc[2], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = b.sequential(*[
            b.ResBlock(nc[3], nc[3], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = b.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = b.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = b.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = b.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[
            b.ResBlock(nc[2], nc[2], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = b.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[
            b.ResBlock(nc[1], nc[1], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = b.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[
            b.ResBlock(nc[0], nc[0], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = b.conv(nc[0], out_nc, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x


class Local_path(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=None, nb=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(Local_path, self).__init__()

        if nc is None:
            nc = [64, 128, 256, 512]
        self.m_head = b.conv(in_nc, nc[0], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = b.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = b.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = b.downsample_strideconv
        elif downsample_mode == 'strideconv122':
            downsample_block = b.downsample_strideconv122
        elif downsample_mode == 'strideconv211':
            downsample_block = b.downsample_strideconv211
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = b.sequential(*[
            b.ResBlock(nc[0], nc[0], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = b.sequential(*[
            b.ResBlock(nc[1], nc[1], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = b.sequential(*[
            b.ResBlock(nc[2], nc[2], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = b.sequential(
            *[b.ResBlock(nc[3], nc[3], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False,
                         mode='C' + act_mode + 'C') for _
              in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = b.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = b.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = b.upsample_convtranspose
        elif upsample_mode == 'convtranspose122':
            upsample_block = b.upsample_convtranspose122
        elif upsample_mode == 'convtranspose211':
            upsample_block = b.upsample_convtranspose211
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = b.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[
            b.ResBlock(nc[2], nc[2], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = b.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[
            b.ResBlock(nc[1], nc[1], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = b.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[
            b.ResBlock(nc[0], nc[0], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False,
                       mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = b.conv(nc[0], out_nc, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        # x = x + x1
        return x