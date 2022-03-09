import numpy as np
import os
import codecs
import torch

from driver import DATA_PATH

def save_np_to_txt(np_input,  txt_name, file_dir=""):
    if file_dir.startswith("/"):
        save_dir = file_dir
    else:
        save_dir = os.path.join(DATA_PATH, file_dir)
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, txt_name)
    assert isinstance(np_input, np.ndarray)
    first_line = ",".join([str(sh) for sh in np_input.shape])
    flat_input = np.reshape(np_input, (-1, np_input.shape[-1]))
    with codecs.open(txt_path, "w", "utf-8") as fw:
        fw.write(first_line)
        fw.write("\n")
        for last_dim_data in flat_input:
            last_data_line = ",".join([str(float(c_data)) for c_data in last_dim_data])
            fw.write(last_data_line)
            fw.write("\n")


def save_torch_to_txt(torch_input,  txt_name, file_dir=""):
    if file_dir.startswith("/"):
        save_dir = file_dir
    else:
        save_dir = os.path.join(DATA_PATH, file_dir)
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, txt_name)
    assert isinstance(torch_input, torch.Tensor)
    first_line = ",".join([str(sh) for sh in torch_input.shape])

    flat_input = torch.reshape(torch_input, (-1, torch_input.shape[-1]))
    with codecs.open(txt_path, "w", "utf-8") as fw:
        fw.write(first_line)
        fw.write("\n")
        for last_dim_data in flat_input:
            last_data_line = ",".join([str(float(c_data)) for c_data in last_dim_data])
            fw.write(last_data_line)
            fw.write("\n")

def load_txt_to_np(txt_name, file_dir="", dtype=np.float32):
    if file_dir.startswith("/"):
        save_dir = file_dir
    else:
        save_dir = os.path.join(DATA_PATH, file_dir)
    txt_path = os.path.join(save_dir, txt_name)
    assert os.path.isfile(txt_path)
    with codecs.open(txt_path, "r", "utf-8") as fr:
        file_lines = fr.read().splitlines()
        first_line = file_lines[0]
        in_shape = tuple([int(fl) for fl in first_line.split(",")])
        datas = []
        for data_line in file_lines[1:]:
            c_data = [float(fl) for fl in data_line.split(",")]
            datas.append(c_data)
    res_np = np.array(datas).reshape(in_shape).astype(dtype)
    return res_np
        
def load_txt_to_torch(txt_name, file_dir="", dtype=torch.float32):
    if file_dir.startswith("/"):
        save_dir = file_dir
    else:
        save_dir = os.path.join(DATA_PATH, file_dir)
    txt_path = os.path.join(save_dir, txt_name)
    assert os.path.isfile(txt_path)
    with codecs.open(txt_path, "r", "utf-8") as fr:
        file_lines = fr.read().splitlines()
        first_line = file_lines[0]
        in_shape = tuple([int(fl) for fl in first_line.split(",")])
        datas = []
        for data_line in file_lines[1:]:
            c_data = [float(fl) for fl in data_line.split(",")]
            datas.append(c_data)
    res_np = torch.Tensor(datas, dtype=torch.float32).reshape(in_shape)
    return res_np



