import torch
import time
import musa_torch_extension
from utils.diff import diff_cpu
from utils.save import save_torch_to_txt

"""
  origin algorithm 782.42ms in py
"""

def make_data():
    seq_len = 200
    batch_size = 64
    feature_num = 512
    hidden_size = 256
    num_layers = 1
    inp = torch.rand([seq_len, batch_size, feature_num])
    bidirectional = False
    module = torch.nn.LSTM(input_size=feature_num, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
    save_torch_to_txt(inp.cpu(), txt_name="lstm_input.txt", file_dir="lstm")
    for i in range(num_layers):
        save_torch_to_txt(getattr(module, "weight_hh_l{}".format(i)), txt_name="lstm_weight_hh_l{}.txt".format(i), file_dir="lstm")
        save_torch_to_txt(getattr(module, "weight_ih_l{}".format(i)), txt_name="lstm_weight_ih_l{}.txt".format(i), file_dir="lstm")
    
        save_torch_to_txt(getattr(module, "bias_hh_l{}".format(i)), txt_name="lstm_bias_hh_l{}.txt".format(i), file_dir="lstm")
        save_torch_to_txt(getattr(module, "bias_ih_l{}".format(i)), txt_name="lstm_bias_ih_l{}.txt".format(i), file_dir="lstm")
        if bidirectional:
            save_torch_to_txt(getattr(module, "weight_hh_l{}_reverse".format(i)), txt_name="lstm_weight_hh_l{}_reverse.txt".format(i), file_dir="lstm")
            save_torch_to_txt(getattr(module, "weight_ih_l{}_reverse".format(i)), txt_name="lstm_weight_ih_l{}_reverse.txt".format(i), file_dir="lstm")
        
            save_torch_to_txt(getattr(module, "bias_hh_l{}_reverse".format(i)), txt_name="lstm_bias_hh_l{}_reverse.txt".format(i), file_dir="lstm")
            save_torch_to_txt(getattr(module, "bias_ih_l{}_reverse".format(i)), txt_name="lstm_bias_ih_l{}_reverse.txt".format(i), file_dir="lstm")
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

    save_torch_to_txt(out_cpu, txt_name="lstm_output.txt", file_dir="lstm")
    save_torch_to_txt(hx_cpu[0], txt_name="lstm_ho.txt", file_dir="lstm")
    save_torch_to_txt(hx_cpu[1], txt_name="lstm_co.txt", file_dir="lstm")
    
    lstm_module.to("mtgpu")

    bt = time.time()
    
    inp = inp.to("mtgpu")
    with torch.autograd.inference_mode(mode=True):
        for i in range(test_count + 1):
            if i == 1:
                bt = time.time()
            out_mtgpu, hx_mtgpu = lstm_module(inp)
    et = time.time()
    print("current mt gpu cost:{:.4f}ms".format((et - bt) / test_count * 1000))
    diff_cpu(out_mtgpu, out_cpu)
    diff_cpu(hx_cpu[0], hx_mtgpu[0])
    diff_cpu(hx_cpu[1], hx_mtgpu[1])

if __name__ == "__main__":
    main()

