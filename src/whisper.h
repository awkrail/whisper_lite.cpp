#ifndef WHISPER_H
#define WHISPER_H

#include "ggml.h"
#include "ggml-cpu.h"

#include <string>
#include <vector>
#include <map>
#include <random>
#include <set>

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30
#define WHISPER_MAX_NODES 4096
#define WHISPER_MAX_DECODERS 8

struct whisper_model_loader;
struct whisper_context;
struct whisper_model;
struct whisper_vocab;
struct whisper_state;
struct whisper_hparams;
struct whisper_filters;
struct whisper_layer_encoder;
struct whisper_layer_decoder;
struct whisper_full_params;
struct whisper_kv_cell;
struct whisper_kv_cache;
struct whisper_mel;
struct whisper_sequence;
struct whisper_token_data;
struct whisper_sched;

typedef int32_t whisper_pos;
typedef int32_t whisper_token;
typedef int32_t whisper_seq_id;

template <typename A, typename B>
struct whisper_pair
{
    A first;
    B second;

    whisper_pair(const A& a, const B& b) : first(a), second(b) {}
    whisper_pair() : first(A()), second(B()) {}
};

enum e_model
{
    MODEL_UNKNOWN,
    MODEL_TINY,
    MODEL_BASE,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
};

static const std::map<e_model, std::string> g_model_name = {
    {MODEL_UNKNOWN, "unknown"},
    {MODEL_TINY, "tiny"},
    {MODEL_BASE, "base"},
    {MODEL_SMALL, "small"},
    {MODEL_MEDIUM, "medium"},
    {MODEL_LARGE, "large"},
};

static const std::map<std::string, std::pair<int, std::string>> g_lang = {
    {"en",
     {
         0,
         "english",
     }},
    {"zh",
     {
         1,
         "chinese",
     }},
    {"de",
     {
         2,
         "german",
     }},
    {"es",
     {
         3,
         "spanish",
     }},
    {"ru",
     {
         4,
         "russian",
     }},
    {"ko",
     {
         5,
         "korean",
     }},
    {"fr",
     {
         6,
         "french",
     }},
    {"ja",
     {
         7,
         "japanese",
     }},
    {"pt",
     {
         8,
         "portuguese",
     }},
    {"tr",
     {
         9,
         "turkish",
     }},
    {"pl",
     {
         10,
         "polish",
     }},
    {"ca",
     {
         11,
         "catalan",
     }},
    {"nl",
     {
         12,
         "dutch",
     }},
    {"ar",
     {
         13,
         "arabic",
     }},
    {"sv",
     {
         14,
         "swedish",
     }},
    {"it",
     {
         15,
         "italian",
     }},
    {"id",
     {
         16,
         "indonesian",
     }},
    {"hi",
     {
         17,
         "hindi",
     }},
    {"fi",
     {
         18,
         "finnish",
     }},
    {"vi",
     {
         19,
         "vietnamese",
     }},
    {"he",
     {
         20,
         "hebrew",
     }},
    {"uk",
     {
         21,
         "ukrainian",
     }},
    {"el",
     {
         22,
         "greek",
     }},
    {"ms",
     {
         23,
         "malay",
     }},
    {"cs",
     {
         24,
         "czech",
     }},
    {"ro",
     {
         25,
         "romanian",
     }},
    {"da",
     {
         26,
         "danish",
     }},
    {"hu",
     {
         27,
         "hungarian",
     }},
    {"ta",
     {
         28,
         "tamil",
     }},
    {"no",
     {
         29,
         "norwegian",
     }},
    {"th",
     {
         30,
         "thai",
     }},
    {"ur",
     {
         31,
         "urdu",
     }},
    {"hr",
     {
         32,
         "croatian",
     }},
    {"bg",
     {
         33,
         "bulgarian",
     }},
    {"lt",
     {
         34,
         "lithuanian",
     }},
    {"la",
     {
         35,
         "latin",
     }},
    {"mi",
     {
         36,
         "maori",
     }},
    {"ml",
     {
         37,
         "malayalam",
     }},
    {"cy",
     {
         38,
         "welsh",
     }},
    {"sk",
     {
         39,
         "slovak",
     }},
    {"te",
     {
         40,
         "telugu",
     }},
    {"fa",
     {
         41,
         "persian",
     }},
    {"lv",
     {
         42,
         "latvian",
     }},
    {"bn",
     {
         43,
         "bengali",
     }},
    {"sr",
     {
         44,
         "serbian",
     }},
    {"az",
     {
         45,
         "azerbaijani",
     }},
    {"sl",
     {
         46,
         "slovenian",
     }},
    {"kn",
     {
         47,
         "kannada",
     }},
    {"et",
     {
         48,
         "estonian",
     }},
    {"mk",
     {
         49,
         "macedonian",
     }},
    {"br",
     {
         50,
         "breton",
     }},
    {"eu",
     {
         51,
         "basque",
     }},
    {"is",
     {
         52,
         "icelandic",
     }},
    {"hy",
     {
         53,
         "armenian",
     }},
    {"ne",
     {
         54,
         "nepali",
     }},
    {"mn",
     {
         55,
         "mongolian",
     }},
    {"bs",
     {
         56,
         "bosnian",
     }},
    {"kk",
     {
         57,
         "kazakh",
     }},
    {"sq",
     {
         58,
         "albanian",
     }},
    {"sw",
     {
         59,
         "swahili",
     }},
    {"gl",
     {
         60,
         "galician",
     }},
    {"mr",
     {
         61,
         "marathi",
     }},
    {"pa",
     {
         62,
         "punjabi",
     }},
    {"si",
     {
         63,
         "sinhala",
     }},
    {"km",
     {
         64,
         "khmer",
     }},
    {"sn",
     {
         65,
         "shona",
     }},
    {"yo",
     {
         66,
         "yoruba",
     }},
    {"so",
     {
         67,
         "somali",
     }},
    {"af",
     {
         68,
         "afrikaans",
     }},
    {"oc",
     {
         69,
         "occitan",
     }},
    {"ka",
     {
         70,
         "georgian",
     }},
    {"be",
     {
         71,
         "belarusian",
     }},
    {"tg",
     {
         72,
         "tajik",
     }},
    {"sd",
     {
         73,
         "sindhi",
     }},
    {"gu",
     {
         74,
         "gujarati",
     }},
    {"am",
     {
         75,
         "amharic",
     }},
    {"yi",
     {
         76,
         "yiddish",
     }},
    {"lo",
     {
         77,
         "lao",
     }},
    {"uz",
     {
         78,
         "uzbek",
     }},
    {"fo",
     {
         79,
         "faroese",
     }},
    {"ht",
     {
         80,
         "haitian creole",
     }},
    {"ps",
     {
         81,
         "pashto",
     }},
    {"tk",
     {
         82,
         "turkmen",
     }},
    {"nn",
     {
         83,
         "nynorsk",
     }},
    {"mt",
     {
         84,
         "maltese",
     }},
    {"sa",
     {
         85,
         "sanskrit",
     }},
    {"lb",
     {
         86,
         "luxembourgish",
     }},
    {"my",
     {
         87,
         "myanmar",
     }},
    {"bo",
     {
         88,
         "tibetan",
     }},
    {"tl",
     {
         89,
         "tagalog",
     }},
    {"mg",
     {
         90,
         "malagasy",
     }},
    {"as",
     {
         91,
         "assamese",
     }},
    {"tt",
     {
         92,
         "tatar",
     }},
    {"haw",
     {
         93,
         "hawaiian",
     }},
    {"ln",
     {
         94,
         "lingala",
     }},
    {"ha",
     {
         95,
         "hausa",
     }},
    {"ba",
     {
         96,
         "bashkir",
     }},
    {"jw",
     {
         97,
         "javanese",
     }},
    {"su",
     {
         98,
         "sundanese",
     }},
    {"yue",
     {
         99,
         "cantonese",
     }},
};

struct whisper_hparams
{
    int32_t n_vocab = 51864;
    int32_t n_audio_ctx = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx = 448;
    int32_t n_text_state = 384;
    int32_t n_text_head = 6;
    int32_t n_text_layer = 4;
    int32_t n_mels = 80;
    int32_t ftype = 1;
    float eps = 1e-5f;
};

struct whisper_token_data
{
    whisper_token id;
    whisper_token tid;

    float p;
    float plog;
    float pt;
    float ptsum;

    int64_t t0;
    int64_t t1;

    float vlen;
};

struct whisper_mel
{
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_batch
{
    int32_t n_tokens;

    whisper_token* token;
    whisper_pos* pos;
    int32_t* n_seq_id;
    whisper_seq_id** seq_id;
    int8_t* logits;
};

struct whisper_filters
{
    int32_t n_mel;
    int32_t n_fft;
    std::vector<float> data;
};

struct whisper_sched
{
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
};

struct whisper_layer_encoder
{
    // encoder.blocks.*.attn_ln
    struct ggml_tensor* attn_ln_0_w;
    struct ggml_tensor* attn_ln_0_b;

    // encoder.blocks.*.attn.out
    struct ggml_tensor* attn_ln_1_w;
    struct ggml_tensor* attn_ln_1_b;

    // encoder.blocks.*.attn.query
    struct ggml_tensor* attn_q_w;
    struct ggml_tensor* attn_q_b;

    // encoder.blocks.*.attn.key
    struct ggml_tensor* attn_k_w;

    // encoder.blocks.*.attn.value
    struct ggml_tensor* attn_v_w;
    struct ggml_tensor* attn_v_b;

    // encoder.blocks.*.mlp_ln
    struct ggml_tensor* mlp_ln_w;
    struct ggml_tensor* mlp_ln_b;

    // encoder.blocks.*.mlp.0
    struct ggml_tensor* mlp_0_w;
    struct ggml_tensor* mlp_0_b;

    // encoder.blocks.*.mlp.2
    struct ggml_tensor* mlp_1_w;
    struct ggml_tensor* mlp_1_b;
};

struct whisper_layer_decoder
{
    // decoder.blocks.*.attn_ln
    struct ggml_tensor *attn_ln_0_w;
    struct ggml_tensor *attn_ln_0_b;

    // decoder.blocks.*.attn.out
    struct ggml_tensor *attn_ln_1_w;
    struct ggml_tensor *attn_ln_1_b;

    // decoder.blocks.*.attn.query
    struct ggml_tensor *attn_q_w;
    struct ggml_tensor *attn_q_b;

    // decoder.blocks.*.attn.key
    struct ggml_tensor *attn_k_w;

    // decoder.blocks.*.attn.value
    struct ggml_tensor *attn_v_w;
    struct ggml_tensor *attn_v_b;

    // decoder.blocks.*.cross_attn_ln
    struct ggml_tensor *cross_attn_ln_0_w;
    struct ggml_tensor *cross_attn_ln_0_b;

    // decoder.blocks.*.cross_attn.out
    struct ggml_tensor *cross_attn_ln_1_w;
    struct ggml_tensor *cross_attn_ln_1_b;

    // decoder.blocks.*.cross_attn.query
    struct ggml_tensor *cross_attn_q_w;
    struct ggml_tensor *cross_attn_q_b;

    // decoder.blocks.*.cross_attn.key
    struct ggml_tensor *cross_attn_k_w;

    // decoder.blocks.*.cross_attn.value
    struct ggml_tensor *cross_attn_v_w;
    struct ggml_tensor *cross_attn_v_b;

    // decoder.blocks.*.mlp_ln
    struct ggml_tensor *mlp_ln_w;
    struct ggml_tensor *mlp_ln_b;

    // decoder.blocks.*.mlp.0
    struct ggml_tensor *mlp_0_w;
    struct ggml_tensor *mlp_0_b;

    // decoder.blocks.*.mlp.2
    struct ggml_tensor *mlp_1_w;
    struct ggml_tensor *mlp_1_b;
};

struct whisper_model
{
    e_model type = MODEL_UNKNOWN;

    whisper_hparams hparams;
    whisper_filters filters;
    
    // encoder.positional_embedding
    struct ggml_tensor* e_pe;

    // encoder.conv1
    struct ggml_tensor *e_conv_1_w;
    struct ggml_tensor *e_conv_1_b;

    // encoder.conv2
    struct ggml_tensor *e_conv_2_w;
    struct ggml_tensor *e_conv_2_b;

    // encoder.ln_post
    struct ggml_tensor *e_ln_w;
    struct ggml_tensor *e_ln_b;

    // decoder.positional_embedding
    struct ggml_tensor *d_pe;

    // decoder.token_embedding
    struct ggml_tensor *d_te;

    // decoder.ln
    struct ggml_tensor *d_ln_w;
    struct ggml_tensor *d_ln_b;

    std::vector<whisper_layer_encoder> layers_encoder;
    std::vector<whisper_layer_decoder> layers_decoder;

    // ggml context that contains all the meta information
    std::vector<ggml_context*> ctxs;
    
    // the model backend data is read-only and can be shared between processors
    std::vector<ggml_backend_buffer_t> buffers;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor*> tensors;
};

struct whisper_vocab
{
    using id = int32_t;
    using token = std::string;

    int n_vocab = 51864;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    id token_eot = 50256;
    id token_sot = 50257; // task tokens (used only for multilingual models)
    id token_translate = 50357;
    id token_transcribe = 50358; // other special tokens
    id token_solm = 50359; // [TDRZ] used by tinydiarize models to indicate speaker turn
    id token_prev = 50360;
    id token_nosp = 50361;
    id token_not = 50362; // no timestamps
    id token_beg = 50363; // begin timestamps
    
    bool is_multilingual() const { return n_vocab >= 51865; }
    
    int num_languages() const { return n_vocab - 51765 - (is_multilingual() ? 1 : 0); }
};

struct whisper_model_loader
{
    void* context;
    size_t (*read)(void* ctx, void* output, size_t read_size);
    bool   (*eof)(void* ctx);
    void  (*close)(void* ctx);
};

struct whisper_kv_cell
{
    whisper_pos pos = -1;
    
    std::set<whisper_seq_id> seq_id;

    bool has_seq_id(const whisper_seq_id& id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

struct whisper_kv_cache
{
    uint32_t head = 0;
    uint32_t size = 0;

    uint32_t n = 0;
    
    std::vector<whisper_kv_cell> cells;

    struct ggml_tensor* k;
    struct ggml_tensor* v;

    ggml_backend_buffer_t buffer = nullptr;

    std::vector<uint8_t> ctx_buf;
};

struct whisper_sequence
{
    std::vector<whisper_token_data> tokens;

    int result_len;

    double sum_logprobs_all;
    double sum_logprobs;
    double avg_logprobs;
    double entropy;
    double score;
};

struct whisper_decoder
{
    whisper_sequence sequence;

    int i_batch;
    int seek_delta;

    bool failed;
    bool completed;
    bool has_ts;

    std::vector<float> probs;
    std::vector<float> logits;
    std::vector<float> logprobs;

    std::vector<whisper_pair<double, whisper_vocab::id>> logits_id;
    mutable std::mt19937 rng;
};

struct whisper_state
{
    int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_batchd_us = 0;
    int64_t t_prompt_us = 0;
    int64_t t_mel_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_encode = 0; // number of encoder calls
    int32_t n_decode =
        0; // number of decoder calls with n_tokens == 1  (text-generation)
    int32_t n_batchd =
        0; // number of decoder calls with n_tokens <  16 (batch decoding)
    int32_t n_prompt =
        0; // number of decoder calls with n_tokens >  1  (prompt encoding)
    int32_t n_fail_p = 0; // number of logprob threshold failures
    int32_t n_fail_h = 0; // number of entropy threshold failures

    // number of decoders for which we have constructed the KV cache
    int32_t kv_self_n_dec = 0;

    // unified self-attention KV cache for all decoders
    whisper_kv_cache kv_self;

    // cross-attention KV cache for the decoders
    // shared between all decoders
    whisper_kv_cache kv_cross;

    // padded buffer for flash-attention
    whisper_kv_cache kv_pad;

    whisper_mel mel;

    whisper_batch batch;

    whisper_decoder decoders[WHISPER_MAX_DECODERS];

    std::vector<ggml_backend_t> backends;

    // - stores meta info about the intermediate tensors into the `meta` buffers
    whisper_sched sched_conv;
    whisper_sched sched_encode;
    whisper_sched sched_cross;
    whisper_sched sched_decode;

    struct ggml_tensor* embd_conv = nullptr;
    struct ggml_tensor* embd_enc = nullptr;

    // helpers for GPU offloading
    std::vector<float> inp_mel;
    std::vector<float> inp_mask;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;

    // std::vector<whisper_segment> result_all;
    int32_t exp_n_audio_ctx = 0; // 0 - use default
};

struct whisper_context
{
    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16;
    ggml_type itype = ggml_type::GGML_TYPE_F16;

    whisper_model model;
    whisper_vocab vocab;

    whisper_state* state = nullptr;
    std::string path_model;
};

struct whisper_full_params
{
    int n_threads;
    int n_max_text_ctx;     // max tokens to use from past text as prompt for the decoder
    int offset_ms;          // start offset in ms
    int duration_ms;        // audio duration to process in ms

    bool translate;
    bool no_context;        // do not use past transcription (if any) as initial prompt for the decoder
    bool no_timestamps;     // do not generate timestamps
    bool single_segment;    // force single segment output (useful for streaming)
    bool print_special;     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
    bool print_progress;    // print progress information
    bool print_realtime;    // print results from within whisper.cpp (avoid it, use callback instead)
    bool print_timestamps;  // print timestamps for each text segment when printing realtime

    // [EXPERIMENTAL] token-level timestamps
    bool  token_timestamps; // enable token-level timestamps
    float thold_pt;         // timestamp token probability threshold (~0.01)
    float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
    int   max_len;          // max segment length in characters
    bool  split_on_word;    // split on word rather than on token (when used with max_len)
    int   max_tokens;       // max tokens per segment (0 = no limit)

    // [EXPERIMENTAL] speed-up techniques
    // note: these can significantly reduce the quality of the output
    bool debug_mode;        // enable debug_mode provides extra info (eg. Dump log_mel)
    int  audio_ctx;         // overwrite the audio context size (0 = use default)

    // [EXPERIMENTAL] [TDRZ] tinydiarize
    bool tdrz_enable;       // enable tinydiarize speaker turn detection

    // A regular expression that matches tokens to suppress
    const char * suppress_regex;

    // tokens to provide to the whisper decoder as initial prompt
    // these are prepended to any existing text context from a previous call
    // use whisper_tokenize() to convert text to tokens
    // maximum of whisper_n_text_ctx()/2 tokens are used (typically 224)
    const char * initial_prompt;
    bool carry_initial_prompt; // if true, always prepend initial_prompt to every decode window (may reduce conditioning on previous text)
    const whisper_token * prompt_tokens;
    int prompt_n_tokens;

    // for auto-detection, set to nullptr, "" or "auto"
    const char * language;
    bool detect_language;

    // common decoding parameters:
    bool suppress_blank; // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
    bool suppress_nst;   // non-speech tokens, ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253

    float temperature;      // initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
    float max_initial_ts;   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
    float length_penalty;   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267

    // fallback parameters
    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
    float temperature_inc;
    float entropy_thold;    // similar to OpenAI's "compression_ratio_threshold"
    float logprob_thold;
    float no_speech_thold;

    struct {
        int best_of;    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L264
    } greedy;

    struct {
        int beam_size;  // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L265

        float patience; // TODO: not implemented, ref: https://arxiv.org/pdf/2204.05424.pdf
    } beam_search;

    // called for every newly generated text segment
    /**
    whisper_new_segment_callback new_segment_callback;
    void * new_segment_callback_user_data;

    // called on each progress update
    whisper_progress_callback progress_callback;
    void * progress_callback_user_data;

    // called each time before the encoder starts
    whisper_encoder_begin_callback encoder_begin_callback;
    void * encoder_begin_callback_user_data;

    // called each time before ggml computation starts
    ggml_abort_callback abort_callback;
    void * abort_callback_user_data;

    // called by each decoder to filter obtained logits
    whisper_logits_filter_callback logits_filter_callback;
    void * logits_filter_callback_user_data;

    const whisper_grammar_element ** grammar_rules;
    size_t                           n_grammar_rules;
    size_t                           i_start_rule;
    float                            grammar_penalty;

    // Voice Activity Detection (VAD) params
    bool         vad;                         // Enable VAD
    const char * vad_model_path;              // Path to VAD model
    **/
};

struct whisper_full_params whisper_full_default_params();

struct whisper_context* whisper_init_from_file_with_params(const char* path_model);

int whisper_full(struct whisper_context* ctx, struct whisper_full_params params, const float* samples, int n_samples);

// Frees all allocated memory
void whisper_free(struct whisper_context* ctx);
void whisper_free_state(struct whisper_state*  state);
// void whisper_free_params(struct whisper_full_params * params);
// void whisper_free_context_params(struct whisper_context_params * params);

#endif
