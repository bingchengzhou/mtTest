import torch
import time
from utils.save import save_torch_to_txt
from utils.diff import diff_cpu

def make_data():
    inp = torch.rand([1, 512, 202, 202])
    module = torch.nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=(2, 2),
    stride=(2, 2), bias=False)
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
    
    inp = inp.to("cuda")
    module.eval()
    module.to("cuda")
    bt = time.time()
    with torch.autograd.inference_mode(mode=True):
        for i in range(test_count + 1):
            if i == 1:
                bt = time.time()
            out_cuda = module(inp)
    torch.cuda.synchronize()
    et = time.time()
    print("current cuda gpu cost:{:.4f}ms".format((et - bt) / test_count * 1000))
    diff_cpu(out_cuda, out_cpu)
    print("first_data:{}, last_data:{}".format(out_cpu[0][0][0][0], out_cpu[-1][-1][-1][-1]))


if __name__ == "__main__":
    main()

