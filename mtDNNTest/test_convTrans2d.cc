#include <CL/cl.h>
#include <cassert>
#include <iostream>
#include <mtdnn.h>
#include <utils.h>
#include <setting.h>


void UserRelease(void* ptr) {
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
  std::string input_file_path(DATA_DIR"convTrans2d/convTrans2d_input.txt");
  std::vector<float> input_data_vec;
  std::vector<int> input_dims;
  readfile(input_dims, input_data_vec, input_file_path);

  std::string weight_file_path(DATA_DIR"convTrans2d/convTrans2d_weight.txt");
  std::vector<float> weight_data_vec;
  std::vector<int> weight_dims;
  readfile(weight_dims, weight_data_vec, weight_file_path);

  ::mt::dnn::Convolution conv;
  mt::dnn::Convolution::Algorithm algo = mt::dnn::Convolution::Algorithm::IMPLICIT_GEMM;
  mt::dnn::Convolution::ComputeMode mode =
      mt::dnn::Convolution::ComputeMode::ALL;
  // set conv info as test_convTrans2d.cc
  const int pad_h = 0, pad_w = 0;
  const int dilation_h = 1, dilation_w = 1;
  const int stride_h = 2, stride_w = 2;
  conv.SetNdInfo({pad_h, pad_w}, {stride_h, stride_w},
                 {dilation_h, dilation_w});
  conv.SetComputeMode(mode);

  const int batch_size = input_dims[0];
  const int channels_in = input_dims[1];
  const int height = input_dims[2];
  const int width = input_dims[3];

  assert(weight_dims[0] == channels_in);
  const int channes_out = weight_dims[1];
  const int kernel_h = weight_dims[2];
  const int kernel_w = weight_dims[3];
  
  const int outHeight = 
      (height - 1) * stride_h- 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int outWidth =
      (width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  cl_mem clmem_input, clmem_w, clmem_output;
  const int dataInSize =
      batch_size * channels_in * height * width * sizeof(float);
  const int dataWSize =
      channels_in * channes_out * kernel_h * kernel_w * sizeof(float);
  const int dataOutSize =
      batch_size * channes_out * outHeight * outWidth * sizeof(float);
  Tensor::Allocate(dataInSize, reinterpret_cast<void **>(&clmem_input));
  Tensor::Allocate(dataWSize, reinterpret_cast<void **>(&clmem_w));
  Tensor::Allocate(dataOutSize, reinterpret_cast<void **>(&clmem_output));

  Tensor::MemcpyH2D(clmem_input, input_data_vec.data(), dataInSize);
  Tensor::MemcpyH2D(clmem_w, weight_data_vec.data(), dataWSize);

  Tensor t_input, t_w, t_output;
  t_input.SetType(Tensor::Type::FLOAT);
  t_input.SetNdInfo({batch_size, channels_in, height, width});
  t_input.SetAddr(clmem_input);

  t_w.SetType(Tensor::Type::FLOAT);
  t_w.SetNdInfo({channels_in, channes_out, kernel_h, kernel_w});
  t_w.SetAddr(clmem_w);

  t_output.SetType(Tensor::Type::FLOAT);
  t_output.SetNdInfo({batch_size, channes_out, outHeight, outWidth});
  t_output.SetAddr(clmem_output);

  ::mt::dnn::Handle h;
  conv.RunBwdData(h, t_output, t_input, t_w, algo, UserAlloc);

  std::vector<float> output(dataOutSize / sizeof(float));
  std::cout << "dataOutSize:" << dataOutSize << std::endl;
  Tensor::MemcpyD2H(output.data(), clmem_output, (size_t)dataOutSize);
  std::cout << "out first data:" << output[0] << ",end data:" << output[dataOutSize / sizeof(float) - 1] << std::endl;
  Tensor::Release(clmem_output);
  Tensor::Release(clmem_input);
  Tensor::Release(clmem_w);
  return 0;
}
