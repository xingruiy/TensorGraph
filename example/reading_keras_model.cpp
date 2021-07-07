#include "tensor_graph.h"
#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s path_to_pb_model\n", argv[0]);
        exit(0);
    }

    TensorGraph graph;
    graph.load_graph(argv[1]);
    graph.create_session();
    graph.set_allow_growth(true);
    graph.set_input_nodes({{"Inputs", 0}});
    graph.set_output_nodes({{"Identity", 0}});

    int64_t dims_in[] = {2, 3};
    int64_t dims_out[] = {2, 1};
    size_t bytes_in = 6 * sizeof(float);
    size_t bytes_out = 2 * sizeof(float);
    std::vector<TF_Tensor *> tensors_in = {
        TF_AllocateTensor(TF_FLOAT, dims_in, 2, bytes_in)};
    std::vector<TF_Tensor *> tensors_out = {
        TF_AllocateTensor(TF_FLOAT, dims_out, 2, bytes_out)};

    float input[6] = {1, 1, 1, 1, 1, 1};
    memcpy(TF_TensorData(tensors_in[0]), input, bytes_in);

    graph.run_session(tensors_in, tensors_out);

    float *output = static_cast<float *>(TF_TensorData(tensors_out[0]));
    std::cout << output[0] << " " << output[1] << std::endl;
}