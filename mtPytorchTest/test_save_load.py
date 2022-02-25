import numpy as np
from torch import dtype
from utils.save import save_np_to_txt, load_txt_to_np


def main():
    test_input = np.random.random(size=(1, 4, 8, 9)).astype(np.float32)
    save_np_to_txt(test_input, "test_save_load.txt")
    res_input = load_txt_to_np(txt_name="test_save_load.txt")
    assert np.all(res_input == test_input)


if __name__ == "__main__":
    main()
