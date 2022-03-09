import torch

def diff_cpu(output_mt, output):
#    print(torch.abs(output_mt.cpu() - output.cpu()))
    diff_res = output_mt.cpu() - output.cpu()
    l1_sum_error = torch.sum(torch.abs(diff_res))
    l1_avg_error = torch.mean(torch.abs(diff_res))
    l1_max_error = torch.max(torch.abs(diff_res))
    print("size", output.size())
    print("l1_sum_error", l1_sum_error)
    print("l1_avg_error", l1_avg_error)
    print("l1_max_error", l1_max_error)
    clamp_diff = torch.clamp(diff_res, -0.000001, 0.000001)
    l1_clamp_sum_error = torch.sum(torch.abs(diff_res - clamp_diff))
    l1_clamp_avg_error = torch.mean(torch.abs(diff_res - clamp_diff))
    print("l1_clamp_sum_error", l1_clamp_sum_error)
    print("l1_clamp_avg_error", l1_clamp_avg_error)
