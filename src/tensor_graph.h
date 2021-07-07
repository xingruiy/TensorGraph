#pragma once
#include <memory>
#include <vector>
#include <tensorflow/c/c_api.h>

namespace internal
{
    struct TensorGraphImpl;
} // namespace internal

class TensorGraph
{
public:
    TensorGraph();

    bool load_graph(std::string graph_path);

    bool void create_session();

    void set_allow_growth(bool allowGrowth);

    void load_custom_operators(std::vector<std::string> dllPath);

    void set_input_nodes(std::vector<std::pair<std::string, int>> name_id_pairs);
    void set_output_nodes(std::vector<std::pair<std::string, int>> name_id_pairs);

    bool run_session(std::vector<TF_Tensor *> &input_values, std::vector<TF_Tensor *> &output_values);

    bool run_session(std::vector<float *> in_values,
                     std::vector<std::vector<int64_t>> in_dims,
                     std::vector<std::vector<int64_t>> out_dims,
                     std::vector<float *> &out_values);

private:
    std::shared_ptr<internal::TensorGraphImpl> impl;
};
