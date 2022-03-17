import torch
import time
import musa_torch_extension
from utils.save import save_torch_to_txt
from utils.diff import diff_cpu

def make_data():
    inp = torch.rand([1, 4, 521, 621])
    # module = torch.nn.ConvTranspose2d(in_channels=512, out_channels=768, kernel_size=(2, 2),
    # stride=(2, 2), bias=False)
    module = torch.nn.ConvTranspose2d(in_channels=4, out_channels=521, kernel_size=(3, 3),
    stride=(1, 1), bias=False)
    # save_torch_to_txt(inp.cpu(), txt_name="convTrans2d_input.txt", file_dir="convTrans2d")
    # save_torch_to_txt(module.weight.data.cpu(), txt_name="convTrans2d_weight.txt", file_dir="convTrans2d")
    print("success save data to test")
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
    # save_torch_to_txt(out_cpu.cpu(), txt_name="convTrans2d_output.txt", file_dir="convTrans2d")  
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
    print("current mg gpu cost:{:.4f}ms".format((et - bt) / test_count * 1000))
    diff_cpu(out_mtgpu, out_cpu)
    print("first_data:{}, last_data:{}".format(out_cpu[0][0][0][0], out_cpu[-1][-1][-1][-1]))


if __name__ == "__main__":
    main()

