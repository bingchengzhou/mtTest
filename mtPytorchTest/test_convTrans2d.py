import torch
import time
import musa_torch_extension
from utils.save import save_torch_to_txt
from utils.diff import diff_mt_cpu

def make_data():
    inp = torch.rand([1, 64, 501, 501])
    module = torch.nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2),
    stride=(2, 2), bias=False)
    # save_torch_to_txt(inp.cpu(), txt_name="convTrans2d_input.txt", file_dir="convTrans2d")
    # save_torch_to_txt(module.weight.data.cpu(), txt_name="convTrans2d_weight.txt", file_dir="convTrans2d")
    return inp, module


def main():

    inp, module = make_data()
    test_count = 1

    module.eval()
    bt = time.time()
    with torch.autograd.inference_mode(mode=True):
        for i in range(test_count + 1):
            if i == 1:
                bt = time.time()
            out_cpu = module(inp)
    et = time.time()
    print("current cpu cost:{:.4f}ms".format((et - bt) / test_count * 1000))
    module.to("mtgpu")

    bt = time.time()
    
    inp = inp.to("mtgpu")
    module.eval()
    module.to("mtgpu")
    with torch.autograd.inference_mode(mode=True):
        for i in range(test_count + 1):
            if i == 1:
                bt = time.time()
            out_mtgpu = module(inp)
    et = time.time()
    print("current mt gpu cost:{:.4f}ms".format((et - bt) / test_count * 1000))
    diff_mt_cpu(out_mtgpu, out_cpu)


if __name__ == "__main__":
    main()

