#include <CL/cl.h>
#include <cassert>
#include <iostream>
#include <mtdnn.h>
#include <setting.h>
#include <utils.h>

void UserRelease(void *ptr) {
  std::cout << "Release: " << ptr << std::endl;
  // delete [] ptr;
  ::mt::dnn::Tensor::Release(ptr);
}

::mt::dnn::MemoryHandler UserAlloc(long unsigned s) {
  // void* ptr = reinterpret_cast<void*>(new char[s]);
  // std::cout << "Alloc: " << ptr << std::endl;
  // return ::mt::dnn::MemoryHandler(ptr, UserRelease);
  cl_mem clmem;
  ::mt::dnn::Tensor::Allocate(s, reinterpret_cast<void **>(&clmem));
  return ::mt::dnn::MemoryHandler(clmem, UserRelease);
}

int main() {
  using ::mt::dnn::Tensor;
  std::string input_file_path(DATA_DIR "batchNorm2d/batchNorm2d_input.txt");
  std::vector<float> input_data_vec;
  std::vector<int> input_dims;
  readfile(input_dims, input_data_vec, input_file_path);

  std::string weight_file_path(DATA_DIR "batchNorm2d/batchNorm2d_weight.txt");
  std::vector<float> weight_data_vec;
  std::vector<int> weight_dims;
  readfile(weight_dims, weight_data_vec, weight_file_path);

  std::string bias_file_path(DATA_DIR "batchNorm2d/batchNorm2d_bias.txt");
  std::vector<float> bias_data_vec;
  std::vector<int> bias_dims;
  readfile(bias_dims, bias_data_vec, bias_file_path);

  std::string running_mean_file_path(
      DATA_DIR "batchNorm2d/batchNorm2d_running_mean.txt");
  std::vector<float> running_mean_data_vec;
  std::vector<int> running_mean_dims;
  readfile(running_mean_dims, running_mean_data_vec, running_mean_file_path);

  std::string running_var_file_path(DATA_DIR
                                    "batchNorm2d/batchNorm2d_running_var.txt");
  std::vector<float> running_var_data_vec;
  std::vector<int> running_var_dims;
  readfile(running_var_dims, running_var_data_vec, running_var_file_path);

  ::mt::dnn::BatchNorm bn;
  bn.SetEpsilon(1e-5);

  // set conv info

  const int batch_size = input_dims[0];
  const int channels_in = input_dims[1];
  const int height = input_dims[2];
  const int width = input_dims[3];

  assert(weight_dims[0] == channels_in && bias_dims[0] == channels_in &&
         running_mean_dims[0] == channels_in &&
         running_var_dims[0] == channels_in);

  cl_mem clmem_input, clmem_weight, clmem_bias, clmem_rm, clmem_rv,
      clmem_output;
  const int dataInSize =
      batch_size * channels_in * height * width * sizeof(float);
  const int dataWSize = channels_in * sizeof(float);
  const int dataBSize = channels_in * sizeof(float);
  const int dataRmSize = channels_in * sizeof(float);
  const int dataRvSize = channels_in * sizeof(float);
  const int dataOutSize = dataInSize;
  Tensor::Allocate(
      dataInSize, reinterpret_cast<void **>(&clmem_input));
  Tensor::Allocate(dataWSize, reinterpret_cast<void **>(&clmem_weight));
  Tensor::Allocate(dataBSize, reinterpret_cast<void **>(&clmem_bias));
  Tensor::Allocate(dataRmSize, reinterpret_cast<void **>(&clmem_rm));
  Tensor::Allocate(dataRvSize, reinterpret_cast<void **>(&clmem_rv));
  Tensor::Allocate(dataOutSize, reinterpret_cast<void **>(&clmem_output));

  Tensor::MemcpyH2D(clmem_input, input_data_vec.data(), dataInSize);
  Tensor::MemcpyH2D(clmem_weight, weight_data_vec.data(), dataWSize);
  Tensor::MemcpyH2D(clmem_bias, bias_data_vec.data(), dataWSize);
  Tensor::MemcpyH2D(clmem_rm, running_mean_data_vec.data(), dataWSize);
  Tensor::MemcpyH2D(clmem_rv, running_var_data_vec.data(), dataWSize);

  Tensor t_input, t_w, t_b, t_rm, t_rv, t_output;
  t_input.SetType(Tensor::Type::FLOAT);
  t_input.SetNdInfo({batch_size, channels_in, height, width});
  t_input.SetAddr(clmem_input);
  t_input.SetFormat(Tensor::Format::NCHW);

  t_w.SetType(Tensor::Type::FLOAT);
  t_w.SetNdInfo({channels_in});
  t_w.SetAddr(clmem_weight);

  t_b.SetType(Tensor::Type::FLOAT);
  t_b.SetNdInfo({channels_in});
  t_b.SetAddr(clmem_bias);

  t_rm.SetType(Tensor::Type::FLOAT);
  t_rm.SetNdInfo({channels_in});
  t_rm.SetAddr(clmem_rm);

  t_rv.SetType(Tensor::Type::FLOAT);
  t_rv.SetNdInfo({channels_in});
  t_rv.SetAddr(clmem_rv);

  t_output.SetType(Tensor::Type::FLOAT);
  t_output.SetNdInfo({batch_size, channels_in, height, width});
  t_output.SetAddr(clmem_output);
  t_output.SetFormat(Tensor::Format::NCHW);

  ::mt::dnn::Handle h;
  bn.RunPure(h, t_output, t_input, t_rm, t_rv, t_w, t_b);

  std::vector<float> output(dataOutSize / sizeof(float));
  std::cout << "dataOutSize:" << dataOutSize << std::endl;
  Tensor::MemcpyD2H(output.data(), clmem_output, (size_t)dataOutSize);
  std::cout << "out first data:" << output[0]
            << ",end data:" << output[dataOutSize / sizeof(float) - 1]
            << std::endl;
  Tensor::Release(clmem_output);
  Tensor::Release(clmem_input);
  Tensor::Release(clmem_weight);
  Tensor::Release(clmem_bias);
  Tensor::Release(clmem_rm);
  Tensor::Release(clmem_rv);
  return 0;
}
