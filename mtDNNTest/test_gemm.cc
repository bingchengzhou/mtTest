#include <CL/cl.h>
#include <cassert>
#include <iostream>
#include <mtdnn.h>
#include <setting.h>
#include <utils.h>

void UserRelease(void *ptr) {
  std::cout << "Release: " << ptr << std::endl;
  delete[] ptr;
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
  std::string input_file_path(DATA_DIR "lstm/lstm_input.txt");
  //shape is 257 * 64 * 512
  std::vector<float> input_data_vec;
  std::vector<int> input_dims;
  readfile(input_dims, input_data_vec, input_file_path);

  std::string weight_file_path(DATA_DIR "lstm/lstm_weight_ih_l0.txt");
  // 1024 * 512
  std::vector<float> weight_data_vec;
  std::vector<int> weight_dims;
  readfile(weight_dims, weight_data_vec, weight_file_path);

  const int input_features = input_dims[2];
  const int seq_len = input_dims[0];
  const int batch_size = input_dims[1];
  const int output_features = weight_dims[0];
  assert(weight_dims[1] == input_features);

  cl_mem cl_output, cl_output_new, cl_input, cl_wi;
  const size_t output_size =
      sizeof(float) * seq_len * batch_size * output_features;
  const size_t input_size =
      sizeof(float) * seq_len * batch_size * input_features;
  const size_t wi_size = sizeof(float) * output_features * input_features;
  
  Tensor::Allocate(output_size, reinterpret_cast<void **>(&cl_output));
  Tensor::Allocate(output_size, reinterpret_cast<void **>(&cl_output_new));

  Tensor::Allocate(input_size, reinterpret_cast<void **>(&cl_input));
  Tensor::MemcpyH2D(cl_input, input_data_vec.data(), input_size);

  Tensor::Allocate(wi_size, reinterpret_cast<void **>(&cl_wi));
  Tensor::MemcpyH2D(cl_wi, weight_data_vec.data(), wi_size);

  Tensor output, input, wi;
  Tensor output_new;

  output.SetType(Tensor::Type::FLOAT);
  output_new.SetType(Tensor::Type::FLOAT);
  input.SetType(Tensor::Type::FLOAT);
  wi.SetType(Tensor::Type::FLOAT);


  output.SetNdInfo({seq_len * batch_size, output_features});
  output_new.SetNdInfo({seq_len * batch_size, output_features});
  input.SetNdInfo({seq_len * batch_size, input_features});
  wi.SetNdInfo({output_features, input_features});

  input.SetAddr(cl_input);
  wi.SetAddr(cl_wi);
  output.SetAddr(cl_output);
  output_new.SetAddr(cl_output_new);

  ::mt::dnn::Dot dot;
  ::mt::dnn::Handle h;
  dot.SetAxes(-1, -1);
  dot.Run(h, output, input, wi);

  ::mt::dnn::MatMul matmul;
  matmul.SetTranspose(false, true);
  matmul.Run(h, output_new, input, wi);


  std::vector<float> output_vec(output_size / sizeof(float));
  Tensor::MemcpyD2H(output_vec.data(), cl_output, output_size);
  std::cout << "output first data:" << output_vec[0] << ",output last data:" 
  << output_vec[output_size / sizeof(float) - 1] << std::endl;;
  Tensor::MemcpyD2H(output_vec.data(), cl_output_new, output_size);
  std::cout << "output first data:" << output_vec[0] << ",output last data:" 
  << output_vec[output_size / sizeof(float) - 1] << std::endl;;

}