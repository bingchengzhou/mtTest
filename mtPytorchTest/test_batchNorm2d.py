
import torch
import time
import musa_torch_extension
from utils.diff import diff_mt_cpu
from utils.save import save_torch_to_txt

def make_data():
    inp = torch.rand([1, 512, 102, 703])
    module = torch.nn.BatchNorm2d(num_features=512)
    save_torch_to_txt(inp.cpu(), txt_name="batchNorm2d_input.txt", file_dir="batchNorm2d")
    save_torch_to_txt(module.weight.data.cpu(), txt_name="batchNorm2d_weight.txt", file_dir="batchNorm2d")
    save_torch_to_txt(module.bias.data.cpu(), txt_name="batchNorm2d_bias.txt", file_dir="batchNorm2d")
    save_torch_to_txt(module.running_mean.data.cpu(), txt_name="batchNorm2d_running_mean.txt", file_dir="batchNorm2d")
    save_torch_to_txt(module.running_var.data.cpu(), txt_name="batchNorm2d_running_var.txt", file_dir="batchNorm2d")
    return inp, module


def main():

    inp, bn = make_data()
    test_count = 4

    bn.eval()
    bt = time.time()

    with torch.autograd.inference_mode(mode=True):
        for i in range(test_count + 1):
            if i == 1:
                bt = time.time()
            out_cpu = bn(inp)
    et = time.time()
    print("current cpu cost:{:.4f}ms".format((et - bt) / test_count * 1000))
    bn.to("mtgpu")

    bt = time.time()
    
    inp = inp.to("mtgpu")
    
    with torch.autograd.inference_mode(mode=True):
        for i in range(test_count + 1):
            if i == 1:
                bt = time.time()
            out_mtgpu = bn(inp)
    et = time.time()
    print("current mt gpu cost:{:.4f}ms".format((et - bt) / test_count * 1000))
    diff_mt_cpu(out_mtgpu, out_cpu)


if __name__ == "__main__":
    main()

