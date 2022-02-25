
import torch
import time
import musa_torch_extension
from utils.diff import diff_mt_cpu

def make_data():
    inp = torch.rand([1, 256, 1100, 1400])
    bn = torch.nn.BatchNorm2d(num_features=256)
    return inp, bn


def main():

    inp, bn = make_data()
    test_count = 10

    bn.eval()

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

