

class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()

        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm()
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = 4

        self.norm_0 = norms.SPADE(fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

class OASIS_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 32
        self.channels = [16 * ch, 16 * ch, 16 * ch, 16 * ch, 8 * ch, 4 * ch, 2 * ch, 1 * ch]
        semantic_nc = 3
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels) - 1):
            self.body.append(
                ResnetBlock_with_SPADE(self.channels[i] + semantic_nc, self.channels[i + 1] + semantic_nc))

        self.fc = nn.Conv2d(513, 16 * ch + semantic_nc, 3, padding=1)
        self.conv_img = nn.Conv2d(self.channels[-1] + semantic_nc, 3, 3, padding=1)

    def forward(self, seg, z=None, act=None):
        assert act is not None

        x = torch.cat((z, F.interpolate(seg, size=z.shape[-2:], mode="nearest")), dim=1)
        x = self.fc(x)

        for i in range(6):
            print("bog")
            _seg = torch.cat((act[-i - 1], F.interpolate(seg, size=act[-i - 1].shape[-2:], mode="nearest")), dim=1)

            x = self.body[i](x, _seg)
            if i < 6 - 1:
                x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

class OASIS_AE(nn.Module):

    def __init__(self):
        super(OASIS_AE, self).__init__()
        self.stn = Affine(input_a=3, input_b=1)
        self.step = int(math.log(512, 2)) - 4
        self.I_encoder = Encoder(3, return_activations=True)
        self.C_t_encoder = Encoder(3, return_activations=True)
        self.oasis = OASIS_Generator()

    def forward(self, I_m, C_t, seg):
        affine_transformed = self.stn(C_t, seg)
        I_feat, I_act = self.I_encoder(I_m, step=self.step)
        C_t_feat, C_t_act = self.C_t_encoder(affine_transformed, step=self.step)

        act = []
        for _I_act, _C_t_act in zip(I_act, C_t_act):
            act.append(torch.cat((_I_act, _C_t_act), dim=1))

        x = torch.cat((I_feat, C_t_feat), dim=1)
        x = self.oasis(seg, x, act)

        return x, affine_transformed


class Encoder(nn.Module):
    def __init__(self, in_channels, return_activations, **kwargs):
        super(Encoder, self).__init__()
        self.return_activations = return_activations

        from_rgb = []
        from_rgb.append(nn.Conv2d(in_channels, 64, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 128, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        from_rgb.append(nn.Conv2d(in_channels, 256, kernel_size=1))
        self.from_rgb = nn.ModuleList(from_rgb)

        modules_down = []
        modules_down.append(Down(64, 128, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(128, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        modules_down.append(Down(256, 256, kernel_size=3, padding=1, **kwargs))
        self.modules_down = nn.ModuleList(modules_down)

    def forward(self, input_batch, step=0, alpha=1.0):
        if self.return_activations:
            activations = []
        else:
            activations = None

        x = self.from_rgb[-(step + 1)](input_batch)
        if self.return_activations:
            activations.append(x)

        if step > 0 and alpha < 1.0:
            residual_x = F.interpolate(input_batch, scale_factor=0.5, mode="bilinear", align_corners=False,
                                       recompute_scale_factor=False)
            residual_x = self.from_rgb[-step](residual_x)
        else:
            residual_x = None

        for module_index in range(-(step + 1), 0, 1):
            x = self.modules_down[module_index](x)

            if module_index == -(step + 1) and residual_x is not None:
                x = (1 - alpha) * residual_x + alpha * x

            if self.return_activations:
                activations.append(x)

        if self.return_activations:
            return x, activations
        else:
            return x

class ConvBlock(nn.Module):
    """
    https://github.com/rosinality/progressive-gan-pytorch/blob/master/model.py#L137
    """

    def __init__(self, in_channel, out_channel, kernel_size, padding, mid_channel=None, conv=nn.Conv2d,
                 normalization=nn.BatchNorm2d, spectral_norm=False, relu_slope=0.01):
        super(ConvBlock, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)

        if mid_channel is None:
            mid_channel = out_channel

        if spectral_norm:
            spectral_norm = nn.utils.spectral_norm
        else:
            spectral_norm = lambda x, *args, **kwargs: x

        self.blocks = nn.Sequential(
            spectral_norm(conv(in_channel, mid_channel, kernel_size=kernel_size[0], padding=padding[0])),
            normalization(out_channel),
            nn.LeakyReLU(relu_slope),
            spectral_norm(conv(mid_channel, out_channel, kernel_size=kernel_size[1], padding=padding[1])),
            normalization(out_channel),
            nn.LeakyReLU(relu_slope)
        )

    def forward(self, x):
        return self.blocks(x)

class Down(nn.Module):
    def __init__(self, *args, pooling=nn.MaxPool2d, block=ConvBlock, **kwargs):
        super(Down, self).__init__()

        self.pool = pooling(2)
        self.block = block(*args, **kwargs)

    def forward(self, *args):
        xs = [self.pool(x) for x in args]
        return self.block(*xs)