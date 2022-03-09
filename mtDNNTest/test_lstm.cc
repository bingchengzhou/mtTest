#include <CL/cl.h>
#include <cassert>
#include <iostream>
#include <mtdnn.h>
#include <setting.h>
#include <utils.h>

void load_weight_bias_and_check(std::vector<std::vector<float>> &weight_vecs,
                                std::vector<std::vector<float>> &bias_vecs,
                                int &hidden_size,
                                bool is_reverse = false,
                                bool is_hidden = false,
                                int layer_num = 0) {
  std::vector<float> weight_vec;
  std::vector<float> bias_vec;
  std::vector<int> weight_dims;
  std::vector<int> bias_dims;
  std::string wh_file_path(DATA_DIR);
  std::string bh_file_path(DATA_DIR);
  bool is_first_layer = layer_num == 0;

  wh_file_path += (is_hidden ? "lstm/lstm_weight_hh_l"
                  : "lstm/lstm_weight_ih_l") + std::to_string(layer_num) + 
                  (is_reverse? "_reverse.txt" : ".txt");
  bh_file_path += (is_hidden ? "lstm/lstm_bias_hh_l"
                  : "lstm/lstm_bias_ih_l") + std::to_string(layer_num) + 
                  (is_reverse ? "_reverse.txt" : ".txt");

  readfile(weight_dims, weight_vec, wh_file_path);
  readfile(bias_dims, bias_vec, bh_file_path);

  if (is_first_layer && !is_hidden) {
    assert(weight_dims.size() == 2 && weight_dims[0] % 4 == 0);
    if (!is_reverse)
      hidden_size = weight_dims[0] / 4;
  } else {
    assert(weight_dims.size() == 2 && weight_dims[0] == 4 * hidden_size &&
           weight_dims[1] == hidden_size);
  }

  assert(bias_dims.size() == 1 && bias_dims[0] == 4 * hidden_size);

  weight_vecs.push_back(weight_vec);
  bias_vecs.push_back(bias_vec);
}

void UserRelease(void *ptr) {
  // std::cout << "Release: " << ptr << std::endl;
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
  std::string input_file_path(DATA_DIR "lstm/lstm_input.txt");
  std::vector<float> input_data_vec;
  std::vector<int> input_dims;
  readfile(input_dims, input_data_vec, input_file_path);

  const int num_layers = 1;
  const int input_features = input_dims[2];
  const int seq_len = input_dims[0];
  const int batch_size = input_dims[1];
  bool bidirectional = false;
  int split_nums = 4; // lstm weight has to spilt to 4 seq.
  int hidden_size;

  std::vector<std::vector<float>> weight_vecs;
  std::vector<std::vector<float>> bias_vecs;
  for (int i = 0; i < num_layers; i++) {
    load_weight_bias_and_check(weight_vecs, bias_vecs, hidden_size, false, false, i);
    load_weight_bias_and_check(weight_vecs, bias_vecs, hidden_size, false, true, i);
    if (bidirectional) {
      load_weight_bias_and_check(weight_vecs, bias_vecs, hidden_size, true, false, i);
      load_weight_bias_and_check(weight_vecs, bias_vecs, hidden_size, true, true, i);
    }
  }
  std::vector<float> hi_vec(seq_len * batch_size * hidden_size, 0.0f);
  std::vector<float> ci_vec(seq_len * batch_size * hidden_size, 0.0f);
  ;

  ::mt::dnn::RNN rnn;
  rnn.SetMode(::mt::dnn::RNN::Mode::LSTM);
  rnn.SetFormat(::mt::dnn::RNN::Format::SEQ_FIRST);
  rnn.SetBiasMode(::mt::dnn::RNN::BiasMode::BOTH);
  rnn.SetDirection(bidirectional ? ::mt::dnn::RNN::Direction::DUAL
                                 : ::mt::dnn::RNN::Direction::SINGLE);
  cl_mem cl_output, cl_ho, cl_co;
  cl_mem cl_input, cl_hi, cl_ci;
  cl_mem cl_wi, cl_bi, cl_wh, cl_bh;
  cl_mem cl_wir, cl_bir, cl_whr, cl_bhr;
  cl_mem tmp_cl_output;
  const size_t output_size =
      sizeof(float) * seq_len * batch_size * hidden_size *
      (bidirectional ? 2 : 1);
  const size_t input_size =
           sizeof(float) * seq_len * batch_size * input_features;
  const size_t hi_size = output_size; 
  const size_t ci_size = output_size;

  size_t wi_size = sizeof(float) * hidden_size * 4 * input_features;
  const size_t bi_size = sizeof(float) * hidden_size * 4;
  const size_t wh_size = sizeof(float) * hidden_size * 4 * hidden_size;
  const size_t bh_size = sizeof(float) * hidden_size * 4;

  Tensor::Allocate(output_size, reinterpret_cast<void **>(&cl_output));
  Tensor::Allocate(output_size, reinterpret_cast<void **>(&cl_ho));
  Tensor::Allocate(output_size, reinterpret_cast<void **>(&cl_co));

  Tensor::Allocate(input_size, reinterpret_cast<void **>(&cl_input));
  Tensor::MemcpyH2D(cl_input, input_data_vec.data(), input_size);

  Tensor::Allocate(hi_size, reinterpret_cast<void **>(&cl_hi));
  Tensor::MemcpyH2D(cl_hi, hi_vec.data(), hi_size);

  Tensor::Allocate(ci_size, reinterpret_cast<void **>(&cl_ci));
  Tensor::MemcpyH2D(cl_ci, ci_vec.data(), ci_size);

  Tensor::Allocate(wi_size, reinterpret_cast<void **>(&cl_wi));
  Tensor::Allocate(bi_size, reinterpret_cast<void **>(&cl_bi));
  Tensor::Allocate(wh_size, reinterpret_cast<void **>(&cl_wh));
  Tensor::Allocate(bh_size, reinterpret_cast<void **>(&cl_bh));

  Tensor output, ho, co;
  Tensor input, hi, ci;
  Tensor wi, bi, wh, bh;
  Tensor wir, bir, whr, bhr;
  Tensor bwdHint;

  output.SetType(Tensor::Type::FLOAT);
  ho.SetType(Tensor::Type::FLOAT);
  co.SetType(Tensor::Type::FLOAT);
  output.SetNdInfo(
      {seq_len, batch_size, hidden_size * (bidirectional ? 2 : 1)});
  ho.SetNdInfo({seq_len, batch_size, hidden_size * (bidirectional ? 2 : 1)});
  co.SetNdInfo({seq_len, batch_size, hidden_size * (bidirectional ? 2 : 1)});

  input.SetType(Tensor::Type::FLOAT);
  input.SetNdInfo({seq_len, batch_size, input_features});
  hi.SetType(Tensor::Type::FLOAT);
  ci.SetType(Tensor::Type::FLOAT);
  hi.SetNdInfo({seq_len, batch_size, hidden_size * (bidirectional ? 2 : 1)});
  ci.SetNdInfo({seq_len, batch_size, hidden_size * (bidirectional ? 2 : 1)});

  wi.SetType(Tensor::Type::FLOAT);
  wi.SetNdInfo({hidden_size * 4, input_features});
  bi.SetType(Tensor::Type::FLOAT);
  bi.SetNdInfo({hidden_size * 4});
  bi.SetType(Tensor::Type::FLOAT);
  wh.SetType(Tensor::Type::FLOAT);
  wh.SetNdInfo({hidden_size * 4, hidden_size});
  bh.SetType(Tensor::Type::FLOAT);
  bh.SetNdInfo({hidden_size * 4});

  const int weights_per_layer = bidirectional ? 4 : 2;
  const int bias_per_layer = bidirectional ? 4 : 2;

  for (int i = 0; i < num_layers; i++) {
    if (i % 2 == 0) {
      if (i == 0) {
        input.SetAddr(cl_input);
      } else {
        input.SetAddr(tmp_cl_output);
      }
      hi.SetAddr(cl_hi);
      ci.SetAddr(cl_hi);

      output.SetAddr(cl_output);
      ho.SetAddr(cl_ho);
      co.SetAddr(cl_co);

    } else {
      if (i == 1) {
        input.SetNdInfo(
            {seq_len, batch_size, hidden_size * (bidirectional ? 2 : 1)});
        Tensor::Allocate(output_size,
                         reinterpret_cast<void **>(&tmp_cl_output));

        wi_size = sizeof(float) * hidden_size * 4 * hidden_size;
        wi.SetNdInfo({hidden_size * 4, hidden_size});
        Tensor::Release(cl_wi);
        Tensor::Allocate(wi_size, reinterpret_cast<void**>(&cl_wi));

        if (bidirectional){
          Tensor::Release(cl_wir);
          Tensor::Allocate(wi_size, reinterpret_cast<void**>(&cl_wir));
        }
      }
      input.SetAddr(cl_output);
      hi.SetAddr(cl_ho);
      ci.SetAddr(cl_co);
      output.SetAddr(tmp_cl_output);
      ho.SetAddr(cl_hi);
      co.SetAddr(cl_ci);
    }
    Tensor::MemcpyH2D(cl_wi, weight_vecs[weights_per_layer * i].data(),
                      wi_size);
    Tensor::MemcpyH2D(cl_wh, weight_vecs[weights_per_layer * i + 1].data(),
                      wh_size);

    Tensor::MemcpyH2D(cl_bi, weight_vecs[weights_per_layer * i].data(),
                      bi_size);
    Tensor::MemcpyH2D(cl_bh, weight_vecs[weights_per_layer * i + 1].data(),
                      bh_size);

    wi.SetAddr(cl_wi);
    wh.SetAddr(cl_wh);
    bi.SetAddr(cl_bi);
    bh.SetAddr(cl_bh);

    if (bidirectional) {
      if (i == 0) {
        Tensor::Allocate(wi_size, reinterpret_cast<void **>(&cl_wir));
        Tensor::Allocate(bi_size, reinterpret_cast<void **>(&cl_bir));
        Tensor::Allocate(wh_size, reinterpret_cast<void **>(&cl_whr));
        Tensor::Allocate(bh_size, reinterpret_cast<void **>(&cl_bhr));
      }
      Tensor::MemcpyH2D(cl_wir, weight_vecs[weights_per_layer * i + 2].data(),
                        wi_size);
      Tensor::MemcpyH2D(cl_whr, weight_vecs[weights_per_layer * i + 3].data(),
                        wh_size);

      Tensor::MemcpyH2D(cl_bir, weight_vecs[weights_per_layer * i + 2].data(),
                        bi_size);
      Tensor::MemcpyH2D(cl_bhr, weight_vecs[weights_per_layer * i + 3].data(),
                        bh_size);

      wir.SetAddr(cl_wir);
      whr.SetAddr(cl_whr);
      bir.SetAddr(cl_bir);
      bhr.SetAddr(cl_bhr);
    }

    ::mt::dnn::Handle h;

    std::vector<Tensor> weight_bias =  {wi, wh, bi, bh};
    if (bidirectional){
      weight_bias.push_back(wir);
      weight_bias.push_back(whr);
      weight_bias.push_back(bir);
      weight_bias.push_back(bhr);
    }

    rnn.RunUnpacked(h, output, ho, co, input, hi, ci, weight_bias.data(), bwdHint,
                  UserAlloc);
  }
  std::vector<float> output_vec(output_size / sizeof(float));
  std::vector<float> ho_vec(output_size / sizeof(float));
  std::vector<float> co_vec(output_size / sizeof(float));
  if (num_layers % 2 == 1) {
    Tensor::MemcpyD2H(output_vec.data(), cl_output, output_size);
    Tensor::MemcpyD2H(ho_vec.data(), cl_ho, output_size);
    Tensor::MemcpyD2H(co_vec.data(), cl_co, output_size);
  } else {
    Tensor::MemcpyD2H(output_vec.data(), tmp_cl_output, output_size);
    Tensor::MemcpyD2H(ho_vec.data(), cl_hi, output_size);
    Tensor::MemcpyD2H(co_vec.data(), cl_ci, output_size);
  }

  Tensor::Release(cl_output);
  Tensor::Release(cl_ho);
  Tensor::Release(cl_co);

  Tensor::Release(cl_input);
  Tensor::Release(cl_hi);
  Tensor::Release(cl_ci);

  Tensor::Release(cl_wi);
  Tensor::Release(cl_bi);
  Tensor::Release(cl_wh);
  Tensor::Release(cl_bh);

  if (num_layers > 1) Tensor::Release(tmp_cl_output);
  if (bidirectional){
    Tensor::Release(cl_wir);
    Tensor::Release(cl_bir);
    Tensor::Release(cl_whr);
    Tensor::Release(cl_bhr);
  }
  
  std::cout << "output first data:" << output_vec[0] << ",output last data:" 
  << output_vec[output_size / sizeof(float) - 1] << std::endl;;
  std::cout << "ho first data:" << ho_vec[0] << ",ho last data:" 
  << ho_vec[output_size / sizeof(float) - 1] << std::endl;
  std::cout << "co first data:" << co_vec[0] << ",co last data:" 
  << co_vec[output_size / sizeof(float) - 1] << std::endl;;
  return 0;
}

