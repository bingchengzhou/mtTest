#include <CL/cl.h>
#include <cassert>
#include <iostream>
#include <mtdnn.h>
#include <utils.h>
#include <setting.h>


void UserRelease(void* ptr) {
  // std::cout << "Release: " << ptr << std::endl;
  // delete [] ptr;
  ::mt::dnn::Tensor::Release(ptr);
}

::mt::dnn::MemoryHandler UserAlloc(long unsigned s) {
  void* ptr = reinterpret_cast<void*>(new char[s]);
  std::cout << "Alloc: " << ptr << std::endl;
  return ::mt::dnn::MemoryHandler(ptr, UserRelease);
  // cl_mem clmem;
  // ::mt::dnn::Tensor::Allocate(s, reinterpret_cast<void **>(&clmem));
  // return ::mt::dnn::MemoryHandler(clmem, UserRelease);
}

int main() {
  using ::mt::dnn::Tensor;
  std::string input_file_path(DATA_DIR"permute/permute_input.txt");
  std::vector<float> input_data_vec;
  std::vector<int> input_dims;
  readfile(input_dims, input_data_vec, input_file_path);


  const int batch_size = input_dims[0];
  const int channels_in = input_dims[1];
  const int height = input_dims[2];
  const int width = input_dims[3];

  cl_mem clmem_input, clmem_output;
  const int dataInSize =
      batch_size * channels_in * height * width * sizeof(float);
  Tensor::Allocate(dataInSize, reinterpret_cast<void **>(&clmem_input));
  Tensor::Allocate(dataInSize, reinterpret_cast<void **>(&clmem_output));

  Tensor::MemcpyH2D(clmem_input, input_data_vec.data(), dataInSize);

  Tensor t_input, t_output;
  t_input.SetType(Tensor::Type::FLOAT);
  t_input.SetNdInfo({batch_size, channels_in, height, width}, {channels_in * height * width, height * width, width, 1});
  t_input.SetAddr(clmem_input);

  t_output.SetType(Tensor::Type::FLOAT);
  t_output.SetNdInfo({batch_size, height, width, channels_in}, {height * width * channels_in, width * channels_in, channels_in, 1});
  t_output.SetAddr(clmem_output);

  ::mt::dnn::Handle h;
  ::mt::dnn::Permute perm;

  perm.Run(h, t_output, t_input);

  std::vector<float> output(dataInSize / sizeof(float));
  std::cout << "dataInSize:" << dataInSize << std::endl;
  Tensor::MemcpyD2H(output.data(), clmem_output, (size_t)dataInSize);
  std::cout << "out first data:" << output[0] << ",end data:" << output[dataInSize / sizeof(float) - 1] << std::endl;
  Tensor::Release(clmem_output);
  Tensor::Release(clmem_input);
  return 0;
}
