from thop import profile
import torch
import musa_torch_extension


def main():
    inp = torch.rand([1, 256, 500, 500])
    module = torch.nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=(2, 2),
    stride=(2, 2), bias=False)
    module.eval()
    with torch.autograd.inference_mode(mode=True):
        macs, params = profile(module, inputs=(inp, ))
    print(macs)
    print(params)


if __name__ == "__main__":
    main()