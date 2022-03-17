#include <CL/cl.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mtdnn.h>
#include <setting.h>
#include <utils.h>

void load_weight_bias_and_check(std::vector<std::vector<float>> &weight_vecs,
                                std::vector<std::vector<float>> &bias_vecs,
                                int &hidden_size, bool is_reverse = false,
                                bool is_hidden = false, int layer_num = 0, int D = 1) {
  std::vector<float> weight_vec;
  std::vector<float> bias_vec;
  std::vector<int> weight_dims;
  std::vector<int> bias_dims;
  std::string wh_file_path(DATA_DIR);
  std::string bh_file_path(DATA_DIR);
  bool is_first_layer = layer_num == 0;

  wh_file_path +=
      (is_hidden ? "bilstm_multilayer/bilstm_weight_hh_l" : "bilstm_multilayer/bilstm_weight_ih_l") +
      std::to_string(layer_num) + (is_reverse ? "_reverse.txt" : ".txt");
  bh_file_path += (is_hidden ? "bilstm_multilayer/bilstm_bias_hh_l" : "bilstm_multilayer/bilstm_bias_ih_l") +
                  std::to_string(layer_num) +
                  (is_reverse ? "_reverse.txt" : ".txt");

  readfile(weight_dims, weight_vec, wh_file_path);
  readfile(bias_dims, bias_vec, bh_file_path);

  if (is_first_layer && !is_hidden) {
    assert(weight_dims.size() == 2 && weight_dims[0] % 4 == 0);
    if (!is_reverse)
      hidden_size = weight_dims[0] / 4;
  } else if (is_hidden){
    assert(weight_dims.size() == 2 && weight_dims[0] == 4 * hidden_size &&
           weight_dims[1] == hidden_size);
  } else{
    assert(weight_dims.size() == 2 && weight_dims[0] == 4 * hidden_size &&
           weight_dims[1] == hidden_size * D);
  }


  assert(bias_dims.size() == 1 && bias_dims[0] == 4 * hidden_size);

  weight_vecs.push_back(weight_vec);
  bias_vecs.push_back(bias_vec);
}

void UserRelease(void *ptr) {
  // delete [] ptr;
  std::cout << "Release: " << ptr << std::endl;
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
  std::string input_file_path(DATA_DIR "bilstm_multilayer/bilstm_input.txt");
  std::vector<float> input_data_vec;
  std::vector<int> input_dims;
  readfile(input_dims, input_data_vec, input_file_path);

  const int num_layers = 4;
  const int input_features = input_dims[2];
  const int seq_len = input_dims[0];
  const int batch_size = input_dims[1];
  bool bidirectional = true;
  int split_nums = 4; // lstm weight has to spilt to 4 seq.
  int D = bidirectional ? 2 : 1;
  int hidden_size;

  std::vector<std::vector<float>> weight_vecs;
  std::vector<std::vector<float>> bias_vecs;
  for (int i = 0; i < num_layers; i++) {
    load_weight_bias_and_check(weight_vecs, bias_vecs, hidden_size, false,
                               false, i, D);
    load_weight_bias_and_check(weight_vecs, bias_vecs, hidden_size, false, true,
                               i, D);
    if (bidirectional) {
      load_weight_bias_and_check(weight_vecs, bias_vecs, hidden_size, true,
                                 false, i, D);
      load_weight_bias_and_check(weight_vecs, bias_vecs, hidden_size, true,
                                 true, i, D);
    }
  }
  std::vector<float> hi_vec(D * num_layers * batch_size * hidden_size, 0.0f);
  std::vector<float> ci_vec(D * num_layers * batch_size * hidden_size, 0.0f);
  const int weights_per_layer = D * 2;
  const int bias_per_layer = D * 2;

  cl_mem cl_output, cl_ho, cl_co;
  cl_mem cl_input, cl_hi, cl_ci;
  cl_mem cl_weights[num_layers * weights_per_layer];
  cl_mem cl_biases[num_layers * bias_per_layer];

  Tensor output, ho, co;
  Tensor input, hi, ci;
  Tensor weights[num_layers * weights_per_layer];
  Tensor biases[num_layers * weights_per_layer];
  Tensor bwdHint;

  const size_t output_size =
      sizeof(float) * seq_len * batch_size * hidden_size * D;
  const size_t input_size =
      sizeof(float) * seq_len * batch_size * input_features;
  const size_t hi_size =
      sizeof(float) * num_layers * D * batch_size * hidden_size;
  const size_t ci_size =
      sizeof(float) * num_layers * D * batch_size * hidden_size;

  Tensor::Allocate(output_size, reinterpret_cast<void **>(&cl_output));
  Tensor::Allocate(hi_size, reinterpret_cast<void **>(&cl_ho));
  Tensor::Allocate(ci_size, reinterpret_cast<void **>(&cl_co));

  Tensor::Allocate(input_size, reinterpret_cast<void **>(&cl_input));
  Tensor::MemcpyH2D(cl_input, input_data_vec.data(), input_size);

  Tensor::Allocate(hi_size, reinterpret_cast<void **>(&cl_hi));
  Tensor::MemcpyH2D(cl_hi, hi_vec.data(), hi_size);

  Tensor::Allocate(ci_size, reinterpret_cast<void **>(&cl_ci));
  Tensor::MemcpyH2D(cl_ci, ci_vec.data(), ci_size);

  output.SetType(Tensor::Type::FLOAT);
  ho.SetType(Tensor::Type::FLOAT);
  co.SetType(Tensor::Type::FLOAT);
  output.SetNdInfo({seq_len, batch_size, hidden_size * D});
  ho.SetNdInfo({D * num_layers, batch_size, hidden_size});
  co.SetNdInfo({D * num_layers, batch_size, hidden_size});
  output.SetAddr(cl_output);
  ho.SetAddr(cl_ho);
  co.SetAddr(cl_co);


  input.SetType(Tensor::Type::FLOAT);
  input.SetNdInfo({seq_len, batch_size, input_features});
  input.SetAddr(cl_input);
  hi.SetType(Tensor::Type::FLOAT);
  ci.SetType(Tensor::Type::FLOAT);
  hi.SetNdInfo({D * num_layers, batch_size, hidden_size});
  ci.SetNdInfo({D * num_layers, batch_size, hidden_size});
  hi.SetAddr(cl_hi);
  ci.SetAddr(cl_ci);

  for (int i = 0; i < num_layers * weights_per_layer; i++) {
    size_t w_size;
    if (i % (weights_per_layer / D) == 0) {
      if (i < weights_per_layer) {
        w_size = sizeof(float) * hidden_size * 4 * input_features;
        weights[i].SetNdInfo({hidden_size * 4, input_features});
      } else {
        w_size = sizeof(float) * hidden_size * 4 * hidden_size * D;
        weights[i].SetNdInfo({hidden_size * 4, hidden_size * D});
      }
    } else {
      w_size = sizeof(float) * hidden_size * 4 * hidden_size;
      weights[i].SetNdInfo({hidden_size * 4, hidden_size});
    }
    weights[i].SetType(Tensor::Type::FLOAT);
    Tensor::Allocate(w_size, reinterpret_cast<void **>(&cl_weights[i]));
    Tensor::MemcpyH2D(cl_weights[i], weight_vecs[i].data(), w_size);
    Tensor::MemcpyD2H(weight_vecs[i].data(), cl_weights[i], w_size);
    weights[i].SetAddr(cl_weights[i]);
  }

  for (int i = 0; i < num_layers * bias_per_layer; i++) {
    size_t b_size = sizeof(float) * hidden_size * 4;
    biases[i].SetType(Tensor::Type::FLOAT);
    biases[i].SetNdInfo({hidden_size * 4});
    Tensor::Allocate(b_size, reinterpret_cast<void **>(&cl_biases[i]));
    Tensor::MemcpyH2D(cl_biases[i], bias_vecs[i].data(), b_size);
    Tensor::MemcpyD2H(bias_vecs[i].data(), cl_biases[i], b_size);
    biases[i].SetAddr(cl_biases[i]);
  }

  std::vector<Tensor> weight_bias;
  for (int i = 0; i < num_layers; i++){
      for (int j = 0; j < D; j++){
        for (int k = 0; k < weights_per_layer / D; k++)
          weight_bias.push_back(weights[i * weights_per_layer + j * weights_per_layer / D + k]);
        for (int k = 0; k < bias_per_layer / D; k++)
          weight_bias.push_back(biases[i * bias_per_layer + j * bias_per_layer / D + k]);
    }
  }

  std::chrono::steady_clock::time_point time_begin =
      std::chrono::steady_clock::now();

  ::mt::dnn::Handle h;
  ::mt::dnn::RNN rnn;
  rnn.SetMode(::mt::dnn::RNN::Mode::LSTM);
  rnn.SetFormat(::mt::dnn::RNN::Format::SEQ_FIRST);
  rnn.SetBiasMode(::mt::dnn::RNN::BiasMode::BOTH);
  rnn.SetDirection(bidirectional ? ::mt::dnn::RNN::Direction::DUAL
                                 : ::mt::dnn::RNN::Direction::SINGLE);
  rnn.SetNumLayers(num_layers);
  
  rnn.RunUnpacked(h, output, ho, co, input, hi, ci, weight_bias.data(), bwdHint,
                  UserAlloc);

  std::chrono::steady_clock::time_point time_end =
      std::chrono::steady_clock::now();
  int time_cost = std::chrono::duration_cast<std::chrono::microseconds>(
                      time_end - time_begin)
                      .count();
  std::cout << "bilstm cost:" << time_cost << "[us]" << std::endl;
  std::vector<float> output_vec(output_size / sizeof(float));
  std::vector<float> ho_vec(hi_size / sizeof(float));
  std::vector<float> co_vec(ci_size / sizeof(float));
  Tensor::MemcpyD2H(output_vec.data(), cl_output, output_size);
  Tensor::MemcpyD2H(ho_vec.data(), cl_ho, hi_size);
  Tensor::MemcpyD2H(co_vec.data(), cl_co, ci_size);

  Tensor::Release(cl_output);
  Tensor::Release(cl_ho);
  Tensor::Release(cl_co);

  Tensor::Release(cl_input);
  Tensor::Release(cl_hi);
  Tensor::Release(cl_ci);

  for (int i = 0; i < num_layers * weights_per_layer; i++) {
    Tensor::Release(cl_weights[i]);
  }
  for (int i = 0; i < num_layers * bias_per_layer; i++) {
    Tensor::Release(cl_biases[i]);
  }

  std::cout << "output first data:" << output_vec[0] << ",output last data:"
            << output_vec[output_size / sizeof(float) - 1] << std::endl;
  ;
  std::cout << "ho first data:" << ho_vec[0]
            << ",ho last data:" << ho_vec[hi_size / sizeof(float) - 1]
            << std::endl;
  std::cout << "co first data:" << co_vec[0]
            << ",co last data:" << co_vec[hi_size / sizeof(float) - 1]
            << std::endl;
  ;
  return 0;
}
