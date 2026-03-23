#include "common-whisper.h"
#include "whisper.h"

#include <cstdio>
#include <string>

struct whisper_params
{
    int32_t n_threads = 4;
    int32_t offset_t_ms = 0;
    int32_t duration_ms = 0;

    float temperature = 0.0f;
    float no_speech_thold = 0.6f;

    std::string language = "en";
    std::string prompt;
    std::string model = "models/ggml-base.en.bin";
    std::string fname_inp;
};

static void print_segments(struct whisper_context * ctx, bool no_timestamps) {
    const int n = whisper_full_n_segments(ctx);
    printf("\n");
    for (int i = 0; i < n; ++i) {
        if (!no_timestamps) {
            printf("[%s --> %s]  ",
                   to_timestamp(0).c_str(),
                   to_timestamp(0).c_str());
        }
        printf("%s\n", whisper_full_get_segment_text(ctx, i));
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "failed to parse args. Usage: ./build/whisper_lite jfk.wav\n");
        return 1;
    }

    whisper_params params;
    params.fname_inp = std::string(argv[1]);

    // init context
    whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str());
    if (!ctx)
    {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 1;
    }

    // load audio data
    std::vector<float> pcmf32;
    if (!read_audio_data(params.fname_inp, pcmf32))
    {
        fprintf(stderr, "error: failed to read '%s'\n", params.fname_inp.c_str());
        whisper_free(ctx);
        return 1;
    }

    // inference params
    whisper_full_params wparams = whisper_full_default_params();

    if (whisper_full(ctx, wparams, pcmf32.data(), (int)pcmf32.size()) != 0)
    {
        fprintf(stderr, "error: failed to process '%s'\n", params.fname_inp.c_str());
        whisper_free(ctx);
        return 1;
    }

    print_segments(ctx, false);

    //whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
