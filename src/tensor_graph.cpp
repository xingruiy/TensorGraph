#include "tensor_graph.h"

struct TensorGraph::TensorGraphImpl
{
    TF_Graph *graph;
    TF_Session *session;
    std::vector<TF_Library *> customOps;
    std::vector<TF_Output> inputs;
    std::vector<TF_Output> outputs;

    inline TensorGraphImpl() : graph(nullptr), session(nullptr)
    {
    }

    ~TensorGraphImpl()
    {
        TF_Status *status = TF_NewStatus();
        TF_DeleteSession(session, status);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);

        for (auto lib : customOps)
        {
            TF_DeleteLibraryHandle(lib);
        }
    }

    static void free_buffer2(void *data, size_t length)
    {
        (void)length; // to suppress warnings
        free(data);
    }

    static void free_buffer(void *data, size_t length, void *arg)
    {
        (void)length; // to suppress warnings
        (void)arg;    // to suppress warnings
        free(data);
    }

    // void set_allow_growth(bool allowGrowth)
    // {
    //     TF_Status *status = TF_NewStatus();
    //     TF_SessionOptions *options = TF_NewSessionOptions();
    //     uint8_t config[11] = {0x32, 0x09, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x3f};
    //     TF_SetConfig(options, (void *)config, 11, status);
    // }

    inline bool load_graph(std::string graph_path)
    {
        TF_Buffer *graph_def = read_file(graph_path);
        TF_Graph *graph_ = TF_NewGraph();

        // import graph_def into graph
        TF_Status *status = TF_NewStatus();
        TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();

        TF_GraphImportGraphDef(graph_, graph_def, opts, status);
        TF_DeleteImportGraphDefOptions(opts);
        if (TF_GetCode(status) != TF_OK)
        {
            fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(status));
            return false;
        }

        fprintf(stdout, "Successfully imported graph\n");

        graph = graph_;
        TF_DeleteStatus(status);
        TF_DeleteBuffer(graph_def);
        return true;
    }

    inline void create_session()
    {
        if (session != nullptr)
            return;

        TF_SessionOptions *sess_opts = TF_NewSessionOptions();
        TF_Status *status = TF_NewStatus();

        // set allow growth
        uint8_t options[] = {0x32, 0x2, 0x20, 0x1};
        TF_SetConfig(sess_opts, options, sizeof(options), status);

        if (TF_GetCode(status) != TF_OK)
        {
            fprintf(stderr, "ERROR: Unable to set session options %s\n", TF_Message(status));
            return;
        }

        TF_DeleteStatus(status);
        status = TF_NewStatus();
        TF_Session *session_ = TF_NewSession(graph, sess_opts, status);

        if (TF_GetCode(status) != TF_OK)
        {
            fprintf(stderr, "ERROR: Unable to create new session %s\n", TF_Message(status));
            return;
        }

        session = session_;
        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(sess_opts);
    }

    inline TF_Buffer *read_file(std::string graph_path)
    {
        FILE *f = fopen(graph_path.c_str(), "rb");
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET); //same as rewind(f);

        void *data = malloc(fsize);
        if (fread(data, fsize, 1, f))
        {
            // to suppress warning
        }

        fclose(f);

        TF_Buffer *buf = TF_NewBuffer();
        buf->data = data;
        buf->length = fsize;
        buf->data_deallocator = TensorGraphImpl::free_buffer2;
        return buf;
    }

    inline void set_input_nodes(std::vector<std::pair<std::string, int>> name_id_pairs)
    {
        std::vector<TF_Output> inputs_;
        for (auto name_id : name_id_pairs)
        {
            auto name = name_id.first.c_str();
            auto id = name_id.second;
            auto op = TF_GraphOperationByName(graph, name);

            if (op == nullptr)
            {
                fprintf(stderr, "ERROR: Unable to retrieve operation %s\n", name);
                return;
            }

            TF_Output input = {op, id};
            inputs_.push_back(input);
        }

        inputs = inputs_;
    }

    inline void set_output_nodes(std::vector<std::pair<std::string, int>> name_id_pairs)
    {
        std::vector<TF_Output> outputs_;
        for (auto name_id : name_id_pairs)
        {
            auto name = name_id.first.c_str();
            auto id = name_id.second;
            auto op = TF_GraphOperationByName(graph, name);

            if (op == nullptr)
            {
                fprintf(stderr, "ERROR: Unable to retrieve operation %s\n", name);
                return;
            }

            TF_Output input = {op, id};
            outputs_.push_back(input);
        }

        outputs = outputs_;
    }

    inline bool run_session(std::vector<TF_Tensor *> &input_values,
                            std::vector<TF_Tensor *> &output_values)
    {
        TF_Status *status = TF_NewStatus();
        TF_SessionRun(
            session, nullptr,
            &inputs[0], &input_values[0], inputs.size(),
            &outputs[0], &output_values[0], outputs.size(),
            nullptr, 0, nullptr, status);

        if (TF_GetCode(status) != TF_OK)
        {
            fprintf(stderr, "ERROR: running session failed %s\n", TF_Message(status));
            return false;
        }

        TF_DeleteStatus(status);
        return true;
    }

    bool run_session(std::vector<float *> in_values,
                     std::vector<std::vector<int64_t>> in_dims,
                     std::vector<std::vector<int64_t>> out_dims,
                     std::vector<float *> &out_values)
    {
        size_t nInput = in_values.size();
        size_t nOutput = out_dims.size();

        std::vector<TF_Tensor *> vTensorsIn;
        std::vector<TF_Tensor *> vTensorsOut;
        vTensorsIn.resize(nInput);
        vTensorsOut.resize(nOutput);
        out_values.resize(nOutput);

        for (size_t i = 0; i < nInput; ++i)
        {
            std::vector<int64_t> dim = in_dims[i];
            int nBytes = sizeof(float);
            int nDims = dim.size();
            for (int j = 0; j < nDims; ++j)
            {
                nBytes *= dim[j];
            }

            vTensorsIn[i] = TF_NewTensor(
                TF_FLOAT, in_dims[i].data(), nDims,
                in_values[i], nBytes, &free_buffer, nullptr);
        }

        for (size_t i = 0; i < nOutput; ++i)
        {
            std::vector<int64_t> dim = out_dims[i];
            int nBytes = sizeof(float);
            int nDims = dim.size();
            for (int j = 0; j < nDims; ++j)
            {
                nBytes *= dim[j];
            }

            vTensorsOut[i] = TF_AllocateTensor(TF_FLOAT, dim.data(), nDims, nBytes);
        }

        if (!run_session(vTensorsIn, vTensorsOut))
            return false;

        for (size_t i = 0; i < nOutput; ++i)
            out_values[i] = static_cast<float *>(TF_TensorData(vTensorsOut[i]));

        return true;
    }

    void load_custom_operators(std::vector<std::string> dllPath)
    {
        for (auto dll : dllPath)
        {
            TF_Status *status = TF_NewStatus();
            auto lib = TF_LoadLibrary(dll.c_str(), status);
            if (TF_GetCode(status) != TF_OK)
            {
                fprintf(stderr, "ERROR: loading custom ops failed %s\n", TF_Message(status));
                TF_DeleteStatus(status);
                return;
            }

            customOps.push_back(lib);
            TF_DeleteStatus(status);
        }
    }
};

TensorGraph::TensorGraph() : impl(new TensorGraphImpl())
{
}

TensorGraph::~TensorGraph() = default;

bool TensorGraph::load_graph(std::string graph_path)
{
    return impl->load_graph(graph_path);
}

void TensorGraph::create_session()
{
    impl->create_session();
}

// void TensorGraph::set_allow_growth(bool allowGrowth)
// {
//     impl->set_allow_growth(allowGrowth);
// }

void TensorGraph::load_custom_operators(std::vector<std::string> dllPath)
{
    impl->load_custom_operators(dllPath);
}

void TensorGraph::set_input_nodes(std::vector<std::pair<std::string, int>> name_id_pairs)
{
    impl->set_input_nodes(name_id_pairs);
}

void TensorGraph::set_output_nodes(std::vector<std::pair<std::string, int>> name_id_pairs)
{
    impl->set_output_nodes(name_id_pairs);
}

bool TensorGraph::run_session(std::vector<TF_Tensor *> &input_values, std::vector<TF_Tensor *> &output_values)
{
    return impl->run_session(input_values, output_values);
}

bool TensorGraph::run_session(std::vector<float *> in_values,
                              std::vector<std::vector<int64_t>> in_dims,
                              std::vector<std::vector<int64_t>> out_dims,
                              std::vector<float *> &out_values)
{
    return impl->run_session(in_values, in_dims, out_dims, out_values);
}