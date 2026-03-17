#include "whisper.h"
#include "whisper-arch.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <fstream>
#include <climits>
#include <cstdarg>
#include <thread>
#include <functional>

static std::string format(const char *fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

static buft_list_t make_buft_list()
{
    buft_list_t buft_list;

    // CPU Extra
    auto *cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto *cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto get_extra_bufts_fn =
        (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
            cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts_fn) {
        ggml_backend_buffer_type_t *extra_bufts = get_extra_bufts_fn(cpu_dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(cpu_dev, *extra_bufts);
            ++extra_bufts;
        }
    }

    // CPU
    buft_list.emplace_back(cpu_dev, ggml_backend_cpu_buffer_type());
    return buft_list;
}

template <typename T>
static void read_safe(whisper_model_loader* loader, T& dest)
{
    loader->read(loader->context, &dest, sizeof(T));
}

const char* whisper_lang_str(int id)
{
    for (const auto &kv : g_lang) {
        if (kv.second.first == id) {
            return kv.first.c_str();
        }
    }
    fprintf(stderr, "%s: unknown language id %d\n", __func__, id);
    return nullptr;
}

static bool weight_buft_supported(const whisper_hparams &hparams,
                                  ggml_tensor *w, ggml_op op,
                                  ggml_backend_buffer_type_t buft,
                                  ggml_backend_dev_t dev)
{
    bool op_supported = true;

    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
        ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU ||
        (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU &&
         buft == ggml_backend_cpu_buffer_type())) {
        // GPU and default CPU backend support all operators
        op_supported = true;
    } else {
        switch (op) {
        // The current extra_buffer_type implementations only support
        // GGML_OP_MUL_MAT and GGML_OP_GET_ROWS
        case GGML_OP_GET_ROWS:
        case GGML_OP_MUL_MAT: {
            ggml_init_params params = {
                /*.mem_size   =*/2 * ggml_tensor_overhead(),
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };

            ggml_context_ptr ctx_ptr{ggml_init(params)};
            if (!ctx_ptr) {
                throw std::runtime_error("failed to create ggml context");
            }
            ggml_context *ctx = ctx_ptr.get();

            ggml_tensor *op_tensor = nullptr;

            if (op == GGML_OP_MUL_MAT) {
                int64_t n_ctx = hparams.n_audio_ctx;
                ggml_tensor *b = ggml_new_tensor_4d(
                    ctx, GGML_TYPE_F32, w->ne[0], n_ctx, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } else if (op == GGML_OP_GET_ROWS) {
                int64_t num_indices = 8;
                ggml_tensor *indices =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_indices);
                op_tensor = ggml_get_rows(ctx, w, indices);
            }

            // create a temporary dummy buffer for the weight so that
            // supports_op can check the buffer type
            GGML_ASSERT(w->buffer == nullptr);
            w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
            op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
            ggml_backend_buffer_free(w->buffer);
            w->buffer = nullptr;
            break;
        }
        default: {
            op_supported = false;
            break;
        }
        };
    }

    return op_supported;
}

static ggml_backend_buffer_type_t
select_weight_buft(const whisper_hparams& hparams, ggml_tensor* w, ggml_op op, buft_list_t buft_list)
{
    for (const auto& p : buft_list)
    {
        ggml_backend_dev_t dev = p.first;
        ggml_backend_buffer_type_t buft = p.second;
        if (weight_buft_supported(hparams, w, op, buft, dev))
        {
            return buft;
        }
    }
    return nullptr;
}

static bool whisper_model_load(struct whisper_model_loader* loader, whisper_context& wctx)
{
    fprintf(stdout, "%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto& model = wctx.model;
    auto& vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC)
        {
            fprintf(stderr, "%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }

    // load hparams
    {
        auto& hparams = model.hparams;

        read_safe(loader, hparams.n_vocab);
        read_safe(loader, hparams.n_audio_ctx);
        read_safe(loader, hparams.n_audio_state);
        read_safe(loader, hparams.n_audio_head);
        read_safe(loader, hparams.n_audio_layer);
        read_safe(loader, hparams.n_text_ctx);
        read_safe(loader, hparams.n_text_state);
        read_safe(loader, hparams.n_text_head);
        read_safe(loader, hparams.n_text_layer);
        read_safe(loader, hparams.n_mels);
        read_safe(loader, hparams.ftype);

        std::string mver = "";

        if (hparams.n_audio_layer == 4)
            model.type = e_model::MODEL_TINY;

        if (hparams.n_audio_layer == 6)
            model.type = e_model::MODEL_BASE;

        if (hparams.n_audio_layer == 12)
            model.type = e_model::MODEL_SMALL;

        if (hparams.n_audio_layer == 24)
            model.type = e_model::MODEL_MEDIUM;

        if (hparams.n_audio_layer == 32)
        {
            model.type = e_model::MODEL_LARGE;

            if (hparams.n_vocab == 51866)
                mver = " v3";
        }

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
        if (wctx.wtype == GGML_TYPE_COUNT) {
            fprintf(stderr, "%s: invalid model (bad ftype value %d)\n", __func__, model.hparams.ftype);
            return false;
        }

        fprintf(stdout, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stdout, "%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        fprintf(stdout, "%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        fprintf(stdout, "%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        fprintf(stdout, "%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        fprintf(stdout, "%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        fprintf(stdout, "%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        fprintf(stdout, "%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        fprintf(stdout, "%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        fprintf(stdout, "%s: n_mels        = %d\n", __func__, hparams.n_mels);
        fprintf(stdout, "%s: ftype         = %d\n", __func__, model.hparams.ftype);
        fprintf(stdout, "%s: qntvr         = %d\n", __func__, qntvr);
        fprintf(stdout, "%s: type          = %d (%s%s)\n", __func__, model.type, g_model_name.at(model.type).c_str(), mver.c_str());
    }

    // load mel filters
    {
        auto& filters = wctx.model.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fft);

        filters.data.resize(filters.n_mel * filters.n_fft);
        loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(loader, n_vocab);

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++)
        {
            uint32_t len;
            read_safe(loader, len);
            
            if (len > 0)
            {
                tmp.resize(len);
                loader->read(loader->context, &tmp[0], tmp.size());
            }
            else
            {
                word = "";
            }
            
            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }

        vocab.n_vocab = model.hparams.n_vocab;

        if (vocab.is_multilingual())
        {
            fprintf(stderr, "%s: Our whisper_lite.cpp supports only English-based models for simplicity.\n", __func__);
            return false;
        }
        
        if (n_vocab < model.hparams.n_vocab)
        {
            fprintf(stdout, "%s: adding %d extra tokens\n", __func__, model.hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < model.hparams.n_vocab; i++)
            {
                if (i > vocab.token_beg) {
                    word = "[_TT_" + std::to_string(i - vocab.token_beg) + "]";
                } else if (i == vocab.token_eot) {
                    word = "[_EOT_]";
                } else if (i == vocab.token_sot) {
                    word = "[_SOT_]";
                } else if (i == vocab.token_translate) {
                    word = "[_TRANSLATE_]";
                } else if (i == vocab.token_transcribe) {
                    word = "[_TRANSCRIBE_]";
                } else if (i == vocab.token_solm) {
                    word = "[_SOLM_]";
                } else if (i == vocab.token_prev) {
                    word = "[_PREV_]";
                } else if (i == vocab.token_nosp) {
                    word = "[_NOSP_]";
                } else if (i == vocab.token_not) {
                    word = "[_NOT_]";
                } else if (i == vocab.token_beg) {
                    word = "[_BEG_]";
                } else if (i > vocab.token_sot &&
                           i <= vocab.token_sot + vocab.num_languages()) {
                    word =
                        "[_LANG_" +
                        std::string(whisper_lang_str(i - vocab.token_sot - 1)) +
                        "]";
                } else {
                    word = "[_extra_token_" + std::to_string(i) + "]";
                }
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }
    }

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;

    const auto& hparams = model.hparams;
    const int n_audio_layer = hparams.n_audio_layer;
    const int n_text_layer = hparams.n_text_layer;

    const size_t n_tensors =
        10 /* input */ + 15 + 15 * n_audio_layer + 24 * n_text_layer;
    
    std::map<ggml_backend_buffer_type_t, ggml_context*> ctx_map;
    auto get_ctx = [&](ggml_backend_buffer_type_t buft) -> ggml_context* {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                n_tensors * ggml_tensor_overhead(),
                nullptr,
                true,
            };

            ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context");
            }

            ctx_map[buft] = ctx;
            model.ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // Create a list of available bufts in priority order
    buft_list_t buft_list = make_buft_list();
    
    auto create_tensor = [&](asr_tensor type, asr_system system, ggml_tensor* meta, int layer = 0) -> ggml_tensor* {
        ggml_op op = ASR_TENSOR_INFO.at(type);
        ggml_backend_buffer_type_t buft = select_weight_buft(hparams, meta, op, buft_list);
        if (!buft)
        {
            throw std::runtime_error(
                format("failed to find a compatible buffer type for tensor %s",
                       ASR_TENSOR_NAMES.at(system).at(type)));
        }

        ggml_context* ctx = get_ctx(buft);
        ggml_tensor* tensor = ggml_dup_tensor(ctx, meta);

        model.tensors[format(ASR_TENSOR_NAMES.at(system).at(type), layer)] =
            tensor;

        return tensor;
    };

    // prepare tensors for the weights
    {
        ggml_init_params params = {
            n_tensors * ggml_tensor_overhead(),
            nullptr,
            true,
        };

        ggml_context* ctx = ggml_init(params);

        const auto& hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;
        const int n_audio_ctx = hparams.n_audio_ctx;
        const int n_audio_state = hparams.n_audio_state;
        const int n_audio_layer = hparams.n_audio_layer;

        const int n_text_ctx = hparams.n_text_ctx;
        const int n_text_state = hparams.n_text_state;
        const int n_text_layer = hparams.n_text_layer;

        const int n_mels = hparams.n_mels;

        model.layers_encoder.resize(n_audio_layer);
        model.layers_decoder.resize(n_text_layer);

        // encoder
        model.e_pe = create_tensor(
            ASR_TENSOR_ENC_POS_EMBD, ASR_SYSTEM_ENCODER,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx));

        model.e_conv_1_w = create_tensor(
            ASR_TENSOR_CONV1_WEIGHT, ASR_SYSTEM_ENCODER,
            ggml_new_tensor_3d(ctx, vtype, 3, n_mels, n_audio_state));
        model.e_conv_1_b = create_tensor(
            ASR_TENSOR_CONV1_BIAS, ASR_SYSTEM_ENCODER,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state));

        model.e_conv_2_w = create_tensor(
            ASR_TENSOR_CONV2_WEIGHT, ASR_SYSTEM_ENCODER,
            ggml_new_tensor_3d(ctx, vtype, 3, n_audio_state, n_audio_state));
        model.e_conv_2_b = create_tensor(
            ASR_TENSOR_CONV2_BIAS, ASR_SYSTEM_ENCODER,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state));

        model.e_ln_w = create_tensor(
            ASR_TENSOR_LN_WEIGHT, ASR_SYSTEM_ENCODER,
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state));
        model.e_ln_b = create_tensor(
            ASR_TENSOR_LN_POST_BIAS, ASR_SYSTEM_ENCODER,
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state));

        for (int i = 0; i < n_audio_layer; ++i) {
            auto &layer = model.layers_encoder[i];

            layer.mlp_ln_w = create_tensor(
                ASR_TENSOR_MLP_LN_WEIGHT, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
            layer.mlp_ln_b = create_tensor(
                ASR_TENSOR_MLP_LN_BIAS, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);

            layer.mlp_0_w =
                create_tensor(ASR_TENSOR_MLP_0_WEIGHT, ASR_SYSTEM_ENCODER,
                              ggml_new_tensor_2d(ctx, wtype, n_audio_state,
                                                 4 * n_audio_state),
                              i);
            layer.mlp_0_b = create_tensor(
                ASR_TENSOR_MLP_0_BIAS, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_audio_state), i);

            layer.mlp_1_w =
                create_tensor(ASR_TENSOR_MLP_2_WEIGHT, ASR_SYSTEM_ENCODER,
                              ggml_new_tensor_2d(ctx, wtype, 4 * n_audio_state,
                                                 n_audio_state),
                              i);
            layer.mlp_1_b = create_tensor(
                ASR_TENSOR_MLP_2_BIAS, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);

            layer.attn_ln_0_w = create_tensor(
                ASR_TENSOR_ATTN_LN_WEIGHT, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
            layer.attn_ln_0_b = create_tensor(
                ASR_TENSOR_ATTN_LN_BIAS, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);

            layer.attn_q_w = create_tensor(
                ASR_TENSOR_ATTN_QUERY_WEIGHT, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state),
                i);
            layer.attn_q_b = create_tensor(
                ASR_TENSOR_ATTN_QUERY_BIAS, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);

            layer.attn_k_w = create_tensor(
                ASR_TENSOR_ATTN_KEY_WEIGHT, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state),
                i);

            layer.attn_v_w = create_tensor(
                ASR_TENSOR_ATTN_VALUE_WEIGHT, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state),
                i);
            layer.attn_v_b = create_tensor(
                ASR_TENSOR_ATTN_VALUE_BIAS, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);

            layer.attn_ln_1_w = create_tensor(
                ASR_TENSOR_ATTN_OUT_WEIGHT, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state),
                i);
            layer.attn_ln_1_b = create_tensor(
                ASR_TENSOR_ATTN_OUT_BIAS, ASR_SYSTEM_ENCODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        }

        // decoder
        model.d_pe = create_tensor(
            ASR_TENSOR_DEC_POS_EMBD, ASR_SYSTEM_DECODER,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_text_state, n_text_ctx));

        model.d_te = create_tensor(
            ASR_TENSOR_DEC_TOKEN_EMBD_WEIGHT, ASR_SYSTEM_DECODER,
            ggml_new_tensor_2d(ctx, wtype, n_text_state, n_vocab));

        model.d_ln_w =
            create_tensor(ASR_TENSOR_LN_WEIGHT, ASR_SYSTEM_DECODER,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state));
        model.d_ln_b =
            create_tensor(ASR_TENSOR_LN_BIAS, ASR_SYSTEM_DECODER,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state));

        for (int i = 0; i < n_text_layer; ++i) {
            auto &layer = model.layers_decoder[i];

            layer.mlp_ln_w = create_tensor(
                ASR_TENSOR_MLP_LN_WEIGHT, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);
            layer.mlp_ln_b = create_tensor(
                ASR_TENSOR_MLP_LN_BIAS, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.mlp_0_w = create_tensor(
                ASR_TENSOR_MLP_0_WEIGHT, ASR_SYSTEM_DECODER,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, 4 * n_text_state),
                i);
            layer.mlp_0_b = create_tensor(
                ASR_TENSOR_MLP_0_BIAS, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_text_state), i);

            layer.mlp_1_w = create_tensor(
                ASR_TENSOR_MLP_2_WEIGHT, ASR_SYSTEM_DECODER,
                ggml_new_tensor_2d(ctx, wtype, 4 * n_text_state, n_text_state),
                i);
            layer.mlp_1_b = create_tensor(
                ASR_TENSOR_MLP_2_BIAS, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.attn_ln_0_w = create_tensor(
                ASR_TENSOR_ATTN_LN_WEIGHT, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);
            layer.attn_ln_0_b = create_tensor(
                ASR_TENSOR_ATTN_LN_BIAS, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.attn_q_w = create_tensor(
                ASR_TENSOR_ATTN_QUERY_WEIGHT, ASR_SYSTEM_DECODER,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state), i);
            layer.attn_q_b = create_tensor(
                ASR_TENSOR_ATTN_QUERY_BIAS, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.attn_k_w = create_tensor(
                ASR_TENSOR_ATTN_KEY_WEIGHT, ASR_SYSTEM_DECODER,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state), i);

            layer.attn_v_w = create_tensor(
                ASR_TENSOR_ATTN_VALUE_WEIGHT, ASR_SYSTEM_DECODER,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state), i);
            layer.attn_v_b = create_tensor(
                ASR_TENSOR_ATTN_VALUE_BIAS, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.attn_ln_1_w = create_tensor(
                ASR_TENSOR_ATTN_OUT_WEIGHT, ASR_SYSTEM_DECODER,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state), i);
            layer.attn_ln_1_b = create_tensor(
                ASR_TENSOR_ATTN_OUT_BIAS, ASR_SYSTEM_DECODER,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.cross_attn_ln_0_w = create_tensor(
                ASR_TENSOR_ATTN_LN_WEIGHT, ASR_SYSTEM_CROSS,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);
            layer.cross_attn_ln_0_b = create_tensor(
                ASR_TENSOR_ATTN_LN_BIAS, ASR_SYSTEM_CROSS,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.cross_attn_q_w = create_tensor(
                ASR_TENSOR_ATTN_QUERY_WEIGHT, ASR_SYSTEM_CROSS,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state), i);
            layer.cross_attn_q_b = create_tensor(
                ASR_TENSOR_ATTN_QUERY_BIAS, ASR_SYSTEM_CROSS,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.cross_attn_k_w = create_tensor(
                ASR_TENSOR_ATTN_KEY_WEIGHT, ASR_SYSTEM_CROSS,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state), i);

            layer.cross_attn_v_w = create_tensor(
                ASR_TENSOR_ATTN_VALUE_WEIGHT, ASR_SYSTEM_CROSS,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state), i);
            layer.cross_attn_v_b = create_tensor(
                ASR_TENSOR_ATTN_VALUE_BIAS, ASR_SYSTEM_CROSS,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);

            layer.cross_attn_ln_1_w = create_tensor(
                ASR_TENSOR_ATTN_OUT_WEIGHT, ASR_SYSTEM_CROSS,
                ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state), i);
            layer.cross_attn_ln_1_b = create_tensor(
                ASR_TENSOR_ATTN_OUT_BIAS, ASR_SYSTEM_CROSS,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state), i);
        }

        ggml_free(ctx);
    }

    // allocate tensors in the backend buffers
    for (auto& p : ctx_map)
    {
        ggml_backend_buffer_type_t buft = p.first;
        ggml_context* ctx = p.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf)
        {
            model.buffers.emplace_back(buf);
            size_t size_main = ggml_backend_buffer_get_size(buf);
            fprintf(stdout, "%s: %12s total size = %8.2f MB\n", __func__, ggml_backend_buffer_name(buf), size_main / 1e6);
        }
    }

    // load weights
    {
        size_t total_size = 0;
        model.n_loaded = 0;

        std::vector<char> read_buf;

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(loader, n_dims);
            read_safe(loader, length);
            read_safe(loader, ttype);
        
            if (loader->eof(loader->context))
                break;

            int32_t nelements = 1;
            int32_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; i++)
            {
                read_safe(loader, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> tmp(length);
            loader->read(loader->context, &tmp[0], tmp.size());
            name.assign(&tmp[0], tmp.size());

            if (model.tensors.find(name) == model.tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                fprintf(stderr, "%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n", __func__, ne[0], ne[1], ne[2], (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2]);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d], expected [%d, %d, %d]\n",
                    __func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }
            
            if (ggml_backend_buffer_is_host(tensor->buffer))
            {
                loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
            } else
            {
                read_buf.resize(ggml_nbytes(tensor));
                loader->read(loader->context, read_buf.data(), read_buf.size());
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }
    
        fprintf(stdout, "%s: model size    = %7.2f MB\n", __func__, total_size / 1e6);

        if (model.n_loaded == 0)
        {
            fprintf(stdout, "%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        }
        else if (model.n_loaded != (int)model.tensors.size())
        {
            fprintf(stderr, "%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }
    
    for (auto &buf : model.buffers)
    {
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    wctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

static std::vector<ggml_backend_t> whisper_backend_init()
{
    std::vector<ggml_backend_t> result;

    // ACCEL backends
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            fprintf(stderr, "%s: using %s backend\n", __func__,
                             ggml_backend_dev_name(dev));
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                fprintf(stderr, "%s: failed to initialize %s backend\n",
                                  __func__, ggml_backend_dev_name(dev));
                continue;
            }
            result.push_back(backend);
        }
    }

    ggml_backend_t backend_cpu =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend_cpu == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    result.push_back(backend_cpu);
    return result;
}

static void whisper_kv_cache_free(struct whisper_kv_cache& cache)
{
    ggml_backend_buffer_free(cache.buffer);
}

static struct whisper_batch whisper_batch_init(int32_t n_tokens,
                                               int32_t n_seq_max)
{
    whisper_batch batch = {
        0, nullptr, nullptr, nullptr, nullptr, nullptr,
    };

    batch.token = (whisper_token *)malloc(sizeof(whisper_token) * (n_tokens));
    batch.pos = (whisper_pos *)malloc(sizeof(whisper_pos) * (n_tokens));
    batch.n_seq_id = (int32_t *)malloc(sizeof(int32_t) * (n_tokens));
    batch.seq_id =
        (whisper_seq_id **)malloc(sizeof(whisper_seq_id *) * (n_tokens + 1));
    for (int i = 0; i < n_tokens; ++i)
    {
        batch.seq_id[i] =
            (whisper_seq_id *)malloc(sizeof(whisper_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens] = nullptr;
    batch.logits = (int8_t *)malloc(sizeof(int8_t) * n_tokens);
    return batch;
}

static void whisper_batch_free(struct whisper_batch& batch)
{
    if (batch.token)
        free(batch.token);
    if (batch.pos)
        free(batch.pos);
    if (batch.n_seq_id)
        free(batch.n_seq_id);
    if (batch.seq_id)
    {
        for (int i = 0; batch.seq_id[i]; ++i)
        {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)
        free(batch.logits);
}

static size_t whisper_sched_size(struct whisper_sched &allocr) {
    size_t size = allocr.meta.size();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(allocr.sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(allocr.sched, i);
        size += ggml_backend_sched_get_buffer_size(allocr.sched, backend);
    }
    return size;
}

static bool whisper_sched_graph_init(struct whisper_sched &allocr, std::vector<ggml_backend_t> backends, std::function<struct ggml_cgraph* ()>&& get_graph)
{
    auto& sched = allocr.sched;
    auto& meta = allocr.meta;

    sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), WHISPER_MAX_NODES, false, true);
    meta.resize(ggml_tensor_overhead() * WHISPER_MAX_NODES + ggml_graph_overhead());
    
    if (!ggml_backend_sched_alloc_graph(sched, get_graph()))
    {
        fprintf(stderr, "%s: failed to allocate the compute buffer\n", __func__);
        return false;
    }

    ggml_backend_sched_reset(sched);
    return true;
}

static struct ggml_cgraph* whisper_build_graph_conv(whisper_context& wctx, whisper_state& wstate)
{
    const auto& model = wctx.model;
    const auto& hparams = model.hparams;
    const int n_ctx = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    GGML_UNUSED(n_state);

    const int n_mels = hparams.n_mels;

    struct ggml_init_params params = {
        wstate.sched_conv.meta.size(),
        wstate.sched_conv.meta.data(),
        true,
    };

    struct ggml_context* ctx0 = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph(ctx0);

    struct ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2 * n_ctx, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor *cur = nullptr;

    // conv + gelu
    {
        cur = ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1);
        cur = ggml_add(ctx0, cur, model.e_conv_1_b);

        cur = ggml_gelu(ctx0, cur);

        cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
        cur = ggml_add(ctx0, cur, model.e_conv_2_b);

        cur = ggml_gelu(ctx0, cur);
    }
    
    ggml_set_name(cur, "embd_conv");
    wstate.embd_conv = cur;

    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph* whisper_build_graph_encoder(whisper_context& wctx, whisper_state& wstate)
{
    const auto& model = wctx.model;
    const auto& hparams = model.hparams;

    const int n_ctx = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head = hparams.n_audio_head;
    const int n_layer = hparams.n_audio_layer;
    const int n_state_head = n_state / n_head;

    auto& kv_pad = wstate.kv_pad;
    const int n_ctx_pad = GGML_PAD(n_ctx, 256);

    struct ggml_init_params params = {
        wstate.sched_encode.meta.size(),
        wstate.sched_encode.meta.data(),
        true,
    };

    struct ggml_context* ctx0 = ggml_init(params);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, WHISPER_MAX_NODES, false);

    struct ggml_tensor* cur = ggml_view_tensor(ctx0, wstate.embd_conv);

    const float KQscale = 1.0f / sqrtf(float(n_state_head));

    // PE
    static int iter = 0;

    const size_t e_pe_stride =
        model.e_pe->ne[0] * ggml_element_size(model.e_pe);
    const size_t e_pe_offset =
        model.e_pe->ne[0] * ggml_element_size(model.e_pe) * n_ctx * iter;

    struct ggml_tensor *e_pe = ggml_view_2d(ctx0, model.e_pe, model.e_pe->ne[0],
                                            n_ctx, e_pe_stride, e_pe_offset);
    cur = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));

    struct ggml_tensor *inpL = cur;
    
    for (int il = 0; il < n_layer; il++)
    {
        const auto& layer = model.layers_encoder[il];

        // norm
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.attn_ln_0_w), layer.attn_ln_0_b);
        }

        // self-attention
        {
            struct ggml_tensor* Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
            Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);

            // note: no bias for Key
            struct ggml_tensor *Kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur);
            struct ggml_tensor *Vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur);
            Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);
            struct ggml_tensor *Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_ctx), 0, 2, 1, 3);

            // flash attn
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, ggml_view_1d(ctx0, kv_pad.k, n_ctx * n_state, 0)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, ggml_view_1d(ctx0, kv_pad.v, n_ctx * n_state, 0)));

            struct ggml_tensor *K = ggml_view_3d(ctx0, kv_pad.k, n_state_head, n_ctx_pad,
                                        n_head, ggml_element_size(kv_pad.k) * n_state,
                                        ggml_element_size(kv_pad.k) * n_state_head, 0);

            struct ggml_tensor *V = ggml_view_3d(ctx0, kv_pad.v, n_state_head, n_ctx_pad,
                                        n_head, ggml_element_size(kv_pad.v) * n_state,
                                        ggml_element_size(kv_pad.v) * n_state_head, 0);

            cur = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f, 0.0f);
            cur = ggml_reshape_2d(ctx0, cur, n_state, n_ctx);
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0, layer.attn_ln_1_w, cur);
            cur = ggml_add(ctx0, cur, layer.attn_ln_1_b);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor* inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);
                cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.mlp_ln_w), layer.mlp_ln_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0, layer.mlp_0_w, cur);
            cur = ggml_add(ctx0, cur, layer.mlp_0_b);

            // GELU
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0, layer.mlp_1_w, cur);
            cur = ggml_add(ctx0, cur, layer.mlp_1_b);
        }
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.e_ln_w), model.e_ln_b);
    }

    ggml_build_forward_expand(gf, cur);
    wstate.embd_enc = cur;
    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph* whisper_build_graph_cross(whisper_context& wctx, whisper_state& wstate)
{
    const auto& model = wctx.model;
    const auto& hparams = model.hparams;

    const int n_ctx = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx
                                                 : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head = hparams.n_audio_head;

    const int n_state_head = n_state / n_head;

    const int n_ctx_pad = GGML_PAD(n_ctx, 256);

    struct ggml_init_params params = {
        /*.mem_size   =*/wstate.sched_cross.meta.size(),
        /*.mem_buffer =*/wstate.sched_cross.meta.data(),
        /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *cur = ggml_view_tensor(ctx0, wstate.embd_enc);

    const float Kscale = pow(float(n_state_head), -0.25);

    for (int il = 0; il < model.hparams.n_text_layer; il++)
    {
        auto& layer = model.layers_decoder[il];

        struct ggml_tensor *Kcross =
            ggml_mul_mat(ctx0, layer.cross_attn_k_w, cur);

        Kcross = ggml_scale(ctx0, Kcross, Kscale);

        struct ggml_tensor *Vcross =
            ggml_mul_mat(ctx0, layer.cross_attn_v_w, cur);

        Vcross = ggml_add(ctx0, Vcross, layer.cross_attn_v_b);

        struct ggml_tensor *k;
        struct ggml_tensor *v;

        k = ggml_view_1d(ctx0, wstate.kv_cross.k, n_state * n_ctx, (ggml_element_size(wstate.kv_cross.k) * n_state) * (il * n_ctx_pad));

        v = ggml_view_1d(ctx0, wstate.kv_cross.v, n_state * n_ctx, (ggml_element_size(wstate.kv_cross.v) * n_state) * (il * n_ctx_pad));

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcross, k));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcross, v)); 
    }

    ggml_free(ctx0);
    return gf;
}

struct whisper_context* whisper_init_with_params_no_state(struct whisper_model_loader* loader)
{
ggml_time_init();

fprintf(stdout, "%s: device = %zu\n", __func__, ggml_backend_dev_count());
    fprintf(stdout, "%s: backends = %zu\n", __func__, ggml_backend_reg_count());

    whisper_context* ctx = new whisper_context;
    if (!whisper_model_load(loader, *ctx))
    {
        loader->close(loader->context);
        fprintf(stderr, "%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }
    loader->close(loader->context);
    return ctx;
}

struct whisper_context* whisper_init_from_file_with_params_no_state(const char* path_model)
{
    fprintf(stdout, "%s: loading model from '%s'\n", __func__, path_model);
    auto fin = std::ifstream(path_model, std::ios::binary);

    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    whisper_model_loader loader = {};

    loader.context = &fin;

    loader.read = [](void* ctx, void* output, size_t read_size) {
        std::ifstream* fin = (std::ifstream*)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void *ctx) {
        std::ifstream *fin = (std::ifstream *)ctx;
        return fin->eof();
    };

    loader.close = [](void *ctx) {
        std::ifstream *fin = (std::ifstream *)ctx;
        fin->close();
    };

    auto ctx = whisper_init_with_params_no_state(&loader);

    if (ctx)
    {
        ctx->path_model = path_model;
    }
    return ctx;
}

static bool whisper_kv_cache_init(struct whisper_kv_cache& cache, ggml_backend_t backend,
                                  ggml_type wtype, int64_t n_text_state, int64_t n_text_layer,
                                  int n_ctx)
{
    const int64_t n_mem = n_text_layer * n_ctx;
    const int64_t n_elements = n_text_state * n_mem;

    cache.ctx_buf.resize(2 * ggml_tensor_overhead());

    struct ggml_init_params params = {
        cache.ctx_buf.size(),
        cache.ctx_buf.data(),
        true,
    };

    cache.head = 0;
    cache.size = n_ctx;

    cache.cells.clear();
    cache.cells.resize(n_ctx);

    struct ggml_context* ctx = ggml_init(params);

    if (!ctx)
    {
        fprintf(stderr,
            "%s: failed to allocate memory for the kv cache context\n",
            __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(ctx, wtype, n_elements);

    cache.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!cache.buffer) {
        fprintf(stderr, "%s: failed to allocate memory for the kv cache\n", __func__);
        return false;
    }

    ggml_backend_buffer_clear(cache.buffer, 0);
    ggml_free(ctx);

    return true;
}

struct whisper_state* whisper_init_state(whisper_context* ctx)
{
    whisper_state* state = new whisper_state;
    state->backends = whisper_backend_init();
    if (state->backends.empty())
    {
        fprintf(stderr, "%s: whisper_backend_init() failed\n", __func__);
        whisper_free_state(state);
        return nullptr;
    }

    state->kv_self_n_dec = 1;
    if (!whisper_kv_cache_init(state->kv_self, state->backends[0], ctx->itype,
                               ctx->model.hparams.n_text_state,
                               ctx->model.hparams.n_text_layer,
                               GGML_PAD(ctx->model.hparams.n_text_ctx, 256)))
    {
        fprintf(stderr,
            "%s: whisper_kv_cache_init() failed for self-attention cache\n",
            __func__);
        whisper_free_state(state);
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->kv_self.k) + ggml_nbytes(state->kv_self.v);
        fprintf(stdout, "%s: kv self size = %7.2f MB\n", __func__, memory_size / 1e6);
    }

    if (!whisper_kv_cache_init(state->kv_cross, state->backends[0], ctx->itype,
                               ctx->model.hparams.n_text_state,
                               ctx->model.hparams.n_text_layer,
                               GGML_PAD(ctx->model.hparams.n_audio_ctx, 256)))
    {
        fprintf(stderr,
                "%s: whisper_kv_cache_init() faild for cross-attention cache\n",
                __func__);
        whisper_free_state(state);
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->kv_cross.k) + ggml_nbytes(state->kv_cross.v);
        fprintf(stdout, "%s: kv cross size = %7.2f MB\n", __func__, memory_size / 1e6);
    }

    if (!whisper_kv_cache_init(state->kv_pad, state->backends[0], ctx->itype,
                               ctx->model.hparams.n_audio_state, 1,
                               GGML_PAD(ctx->model.hparams.n_audio_ctx, 256)))
    {
        fprintf(stderr,
                "%s: whisper_kv_cache_init() faild for self-attention cache\n",
                __func__);
        whisper_free_state(state);
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->kv_pad.k) + ggml_nbytes(state->kv_pad.v);
        fprintf(stdout, "%s: kv pad  size  = %7.2f MB\n", __func__, memory_size / 1e6);
    }

    state->logits.reserve(ctx->vocab.n_vocab * ctx->model.hparams.n_text_ctx);

    state->batch = whisper_batch_init(ctx->model.hparams.n_text_ctx, WHISPER_MAX_DECODERS);

    // TAGS: WHISPER_DECODER_INIT
    state->decoders[0].probs.reserve(ctx->vocab.n_vocab);
    state->decoders[0].logits.reserve(ctx->vocab.n_vocab);
    state->decoders[0].logprobs.reserve(ctx->vocab.n_vocab);
    state->decoders[0].logits_id.reserve(ctx->model.hparams.n_vocab);
    state->decoders[0].rng = std::mt19937(0);

    // conv
    {
        bool ok = whisper_sched_graph_init(state->sched_conv, state->backends, [&]() {
            return whisper_build_graph_conv(*ctx, *state);
        });

        if (!ok)
        {
            fprintf(stderr, "%s: failed to init conv allocator\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }

        fprintf(stdout, "%s: compute buffer (conv) = %7.2f MB\n", __func__, whisper_sched_size(state->sched_conv) / 1e6);
    }

    // encoder
    {
        bool ok = whisper_sched_graph_init(state->sched_encode, state->backends, [&]() {
            return whisper_build_graph_encoder(*ctx, *state);
        });
        
        if (!ok)
        {
            fprintf(stderr, "%s: failed to init encoder allocator\n",
                              __func__);
            whisper_free_state(state);
            return nullptr;
        }

        fprintf(stdout, "%s: compute buffer (encode) = %7.2f MB\n", __func__, whisper_sched_size(state->sched_encode) / 1e6);
    }

    // cross-attn
    {
        bool ok = whisper_sched_graph_init(state->sched_cross, state->backends, [&](){ return whisper_build_graph_cross(*ctx, *state); });

        if (!ok)
        {
            fprintf(stderr, "%s: failed to init cross allocator\n",
                              __func__);
            whisper_free_state(state);
            return nullptr;
        }

        fprintf(stdout, "%s: compute buffer (cross)  = %7.2f MB\n", __func__, whisper_sched_size(state->sched_cross) / 1e6);
    }

    // decoder
    {
    }

    return state;
}

struct whisper_context* whisper_init_from_file_with_params(const char* path_model)
{
    whisper_context* ctx = whisper_init_from_file_with_params_no_state(path_model);
    if (!ctx)
        return nullptr;
    
    ctx->state = whisper_init_state(ctx);
    if (!ctx)
    {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct whisper_full_params whisper_full_default_params()
{
    struct whisper_full_params result = {
        /*.n_threads         =*/
        std::min(4, (int32_t)std::thread::hardware_concurrency()),
        /*.n_max_text_ctx    =*/16384,
        /*.offset_ms         =*/0,
        /*.duration_ms       =*/0,

        /*.translate         =*/false,
        /*.no_context        =*/true,
        /*.no_timestamps     =*/false,
        /*.single_segment    =*/false,
        /*.print_special     =*/false,
        /*.print_progress    =*/false,
        /*.print_realtime    =*/false,
        /*.print_timestamps  =*/true,

        /*.token_timestamps  =*/false,
        /*.thold_pt          =*/0.01f,
        /*.thold_ptsum       =*/0.01f,
        /*.max_len           =*/0,
        /*.split_on_word     =*/false,
        /*.max_tokens        =*/0,

        /*.debug_mode        =*/false,
        /*.audio_ctx         =*/0,

        /*.tdrz_enable       =*/false,

        /* suppress_regex    =*/nullptr,

        /*.initial_prompt       =*/nullptr,
        /*.carry_initial_prompt =*/false,
        /*.prompt_tokens        =*/nullptr,
        /*.prompt_n_tokens      =*/0,

        /*.language          =*/"en",
        /*.detect_language   =*/false,

        /*.suppress_blank    =*/true,
        /*.suppress_nst      =*/false,

        /*.temperature       =*/0.0f,
        /*.max_initial_ts    =*/1.0f,
        /*.length_penalty    =*/-1.0f,

        /*.temperature_inc   =*/0.2f,
        /*.entropy_thold     =*/2.4f,
        /*.logprob_thold     =*/-1.0f,
        /*.no_speech_thold   =*/0.6f,

        /*.greedy            =*/
        {
            /*.best_of   =*/-1,
        },

        /*.beam_search      =*/
        {
            /*.beam_size =*/-1,

            /*.patience  =*/-1.0f,
        },

        ///*.new_segment_callback           =*/nullptr,
        ///*.new_segment_callback_user_data =*/nullptr,

        ///*.progress_callback           =*/nullptr,
        ///*.progress_callback_user_data =*/nullptr,

        ///*.encoder_begin_callback           =*/nullptr,
        ///*.encoder_begin_callback_user_data =*/nullptr,

        ///*.abort_callback                   =*/nullptr,
        ///*.abort_callback_user_data         =*/nullptr,

        ///*.logits_filter_callback           =*/nullptr,
        ///*.logits_filter_callback_user_data =*/nullptr,

        ///*.grammar_rules   =*/nullptr,
        ///*.n_grammar_rules =*/0,
        ///*.i_start_rule    =*/0,
        ///*.grammar_penalty =*/100.0f,

        ///*.vad                         =*/false,
        ///*.vad_model_path              =*/nullptr,

        ///* vad_params =*/whisper_vad_default_params(),
    };
    result.greedy = { 5 };
    return result;
}

int whisper_full_with_state(struct whisper_context* ctx, struct whisper_state* state, struct whisper_full_params params, const float* samples, int n_samples)
{
    return 0;
}

int whisper_full(struct whisper_context* ctx, struct whisper_full_params params, const float* samples, int n_samples)
{
    return whisper_full_with_state(ctx, ctx->state, params, samples, n_samples);
}

void whisper_free_state(struct whisper_state* state)
{
    if (state)
    {
        whisper_kv_cache_free(state->kv_self);
        whisper_kv_cache_free(state->kv_cross);
        whisper_kv_cache_free(state->kv_pad);
        whisper_batch_free(state->batch);

        ggml_backend_sched_free(state->sched_conv.sched);
        ggml_backend_sched_free(state->sched_encode.sched);
        ggml_backend_sched_free(state->sched_cross.sched);
        ggml_backend_sched_free(state->sched_decode.sched);

        for (auto &backend : state->backends) {
            ggml_backend_free(backend);
        }

        delete state;
    }
}

void whisper_free(struct whisper_context* ctx)
{
    if (ctx)
    {
        for (ggml_context* context : ctx->model.ctxs)
        {
            ggml_free(context);
        }

        for (ggml_backend_buffer_t buf : ctx->model.buffers)
        {
            ggml_backend_buffer_free(buf);
        }

        whisper_free_state(ctx->state);
        delete ctx;
    }
}
