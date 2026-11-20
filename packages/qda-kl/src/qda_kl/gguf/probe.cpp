// Pinned libllama bridge: inspect the loaded tensors and expose teacher-forced logits.
#include "llama-model.h"
#include "ggml-backend.h"
#include <cstdio>
#include <dlfcn.h>

extern "C" llama_model * llama_model_load_from_file(const char * path, llama_model_params params) {
    using load_fn = llama_model * (*)(const char *, llama_model_params);
    auto load = reinterpret_cast<load_fn>(dlsym(RTLD_NEXT, "llama_model_load_from_file"));
    if (!load) return nullptr;
    auto model = load(path, params);
    if (model) for (const auto & item : model->tensors_by_name) {
        auto t = item.second;
        std::fprintf(stderr, "QDA_TENSOR\t%s\t%d\t%lld\t%lld\t%lld\t%lld\n", item.first.c_str(),
            int(t->type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    }
    return model;
}

extern "C" {
void * qda_load(const char * path, int gpu_layers) {
    ggml_backend_load_all();
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = gpu_layers;
    params.check_tensors = true;
    params.use_extra_bufts = false;
    params.load_mtp = false;
    return llama_model_load_from_file(path, params);
}
int qda_vocab(llama_model * model) { return llama_vocab_n_tokens(llama_model_get_vocab(model)); }
int qda_tokenize(llama_model * model, const char * text, int length, int * tokens, int count) {
    return llama_tokenize(llama_model_get_vocab(model), text, length, tokens, count, false, true);
}
void * qda_context(llama_model * model, int context, int batch, int threads) {
    auto params = llama_context_default_params();
    params.n_ctx = context;
    params.n_batch = params.n_ubatch = batch;
    params.n_threads = params.n_threads_batch = threads;
    params.type_k = params.type_v = GGML_TYPE_BF16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    return llama_init_from_model(model, params);
}
void qda_clear(llama_context * ctx) { llama_memory_clear(llama_get_memory(ctx), true); }
int qda_decode(llama_context * ctx, const int * tokens, int count, int offset, int start) {
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = offset + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = offset + i >= start;
    }
    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc;
}
float * qda_logits(llama_context * ctx, int index) { return llama_get_logits_ith(ctx, index); }
void qda_free_context(llama_context * ctx) { llama_free(ctx); }
void qda_free_model(llama_model * model) { llama_model_free(model); }
}
