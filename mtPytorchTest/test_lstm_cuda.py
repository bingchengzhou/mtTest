import torch
import time
from utils.diff import diff_cpu

def make_data():
    seq_len = 200
    batch_size = 64
    feature_num = 512
    hidden_size = 256
    num_layers = 1
    inp = torch.rand([seq_len, batch_size, feature_num])
    bidirectional = False
    module = torch.nn.LSTM(input_size=feature_num, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
    return inp, module


def main():

    inp, lstm_module = make_data()
    test_count = 1

    lstm_module.eval()
    bt = time.time()

    with torch.autograd.inference_mode(mode=True):
        for i in range(test_count + 1):
            if i == 1:
                bt = time.time()
            out_cpu, hx_cpu = lstm_module(inp)
    et = time.time()
    print("current cpu cost:{:.4f}ms".format((et - bt) / test_count * 1000))

    lstm_module.to("cuda:0")

    bt = time.time()
    
    inp = inp.to("cuda:0")
    
    with torch.autograd.inference_mode(mode=True):
        for i in range(test_count + 1):
            if i == 1:
                bt = time.time()
            out_cuda, hx_cuda = lstm_module(inp)
            torch.cuda.synchronize()
    et = time.time()
    print("current cuda cost:{:.4f}ms".format((et - bt) / test_count * 1000))
    diff_cpu(out_cpu, out_cuda)
    diff_cpu(hx_cpu[0], hx_cuda[0])
    diff_cpu(hx_cpu[1], hx_cuda[1])

if __name__ == "__main__":
    main()

