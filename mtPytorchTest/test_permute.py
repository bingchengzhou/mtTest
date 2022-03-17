import torch
import time
from utils.save import save_torch_to_txt
from utils.diff import diff_cpu

def make_data():
    inp = torch.rand([3, 16, 45, 32])
    # save_torch_to_txt(inp.cpu(), txt_name="convTrans2d_input.txt", file_dir="convTrans2d")
    save_torch_to_txt(inp.cpu(), txt_name="permute_input.txt", file_dir="permute")
    print("success save data to test")
    return inp


def main():

    inp = make_data()
    output = inp.permute([0, 2, 3, 1])
    save_torch_to_txt(output.cpu(), txt_name="permute_output.txt", file_dir="permute")


if __name__ == "__main__":
    main()

