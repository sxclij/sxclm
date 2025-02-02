#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUFFER_BITSIZE 1024
#define LAYER_BITSIZE 512
#define LAYER_DEPTH 4
#define PARAM_BITSIZE (LAYER_BITSIZE * LAYER_BITSIZE * (LAYER_DEPTH - 1) * 2)
#define EXPLORE_RATE 100
#define MUTATION_RATE 0.01

struct bitset {
    uint64_t* data;
    int32_t size;
};

struct nnmodel {
    struct bitset* param;
    struct bitset* backup;
    struct bitset* state;
    struct bitset* buf;
    const char* teacher;
    char* output;
    int32_t bestscore;
};

void file_read(char* dst, const char* filename) {
    FILE* file = fopen(filename, "r");
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    size_t bytes_read = fread(dst, 1, file_size, file);
    fclose(file);
    dst[file_size] = '\0';
}

struct bitset bitset_init(uint64_t* data, int32_t size) {
    return (struct bitset){.data = data, .size = size};
}

void bitset_set0(struct bitset* bitset, int32_t index) {
    bitset->data[index / 64] &= ~(1ULL << (index % 64));
}

void bitset_set1(struct bitset* bitset, int32_t index) {
    bitset->data[index / 64] |= (1ULL << (index % 64));
}

void bitset_toggle(struct bitset* bitset, int32_t index) {
    bitset->data[index / 64] ^= (1ULL << (index % 64));
}

int bitset_get(const struct bitset* bitset, int32_t index) {
    return (bitset->data[index / 64] & (1ULL << (index % 64))) != 0;
}

void bitset_clear(struct bitset* bitset) {
    memset(bitset->data, 0, bitset->size / 8);
}

void bitset_swap(struct bitset** a, struct bitset** b) {
    struct bitset* t = *a;
    *a = *b;
    *b = t;
}

static inline __m256i popcount256(__m256i v) {
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i popcnt = _mm256_add_epi8(popcnt1, popcnt2);
    __m256i sum = _mm256_sad_epu8(popcnt, _mm256_setzero_si256());
    return sum;
}

static inline int64_t horizontal_sum(__m256i v) {
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi64(vlow, vhigh);
    int64_t res = _mm_cvtsi128_si64(sum128);
    res += _mm_extract_epi64(sum128, 1);
    return res;
}

static int compute_dot(const uint64_t* state, const uint64_t* param, int start_bit) {
    int words_per_block = LAYER_BITSIZE / 64;

    const uint64_t* param_pos = param + (start_bit / 64);
    const uint64_t* param_neg = param + (start_bit / 64) + words_per_block;

    __m256i dot_sum = _mm256_setzero_si256();

    int blocks = words_per_block / 4;
    for (int j = 0; j < blocks; j++) {
        __m256i s = _mm256_loadu_si256((__m256i const*)(state + j * 4));
        __m256i p_pos = _mm256_loadu_si256((__m256i const*)(param_pos + j * 4));
        __m256i p_neg = _mm256_loadu_si256((__m256i const*)(param_neg + j * 4));
        __m256i masked_pos = _mm256_and_si256(s, p_pos);
        __m256i masked_neg = _mm256_and_si256(s, p_neg);
        __m256i cnt_pos = popcount256(masked_pos);
        __m256i cnt_neg = popcount256(masked_neg);
        __m256i diff = _mm256_sub_epi64(cnt_pos, cnt_neg);
        dot_sum = _mm256_add_epi64(dot_sum, diff);
    }
    int dot = (int)horizontal_sum(dot_sum);
    return dot;
}

char nnmodel_calc_ch(struct nnmodel* model, char ch) {
    for (int32_t i = 0; i < 256; i++) {
        bitset_set0(model->state, i + 128);
    }
    bitset_set1(model->state, ch);

    int32_t param_i = 0;
    int8_t maxchar_index = 0;
    int8_t maxchar_value = -128;

    for (int32_t output_i = 0; output_i < LAYER_BITSIZE; output_i++) {
        int dot = compute_dot(model->state->data, model->param->data, param_i);
        param_i += 2 * LAYER_BITSIZE;

        if (dot > maxchar_value && output_i < 256) {
            maxchar_index = output_i;
            maxchar_value = dot;
        }
        if (dot > 0) {
            bitset_set1(model->buf, output_i);
        } else {
            bitset_set0(model->buf, output_i);
        }
    }
    bitset_swap(&model->state, &model->buf);
    return maxchar_index - 128;
}

void nnmodel_calc(struct nnmodel* model, const char* input, int32_t count) {
    int32_t input_size = strlen(input);
    memcpy(model->output, input, input_size);
    bitset_clear(model->state);
    for (int32_t i = input_size; i < input_size + count; i++) {
        model->output[i] = nnmodel_calc_ch(model, input[i]);
    }
    model->output[input_size + count] = '\0';
}

void nnmodel_backup_save(struct nnmodel* model) {
    memcpy(model->backup->data, model->param->data, model->param->size / 8);
}

void nnmodel_backup_load(struct nnmodel* model) {
    memcpy(model->param->data, model->backup->data, model->param->size / 8);
}

static uint32_t xorshift_state = 34563;

uint32_t xorshift(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

void nnmodel_mutation(struct nnmodel* model) {
    int32_t num_mutations = (model->param->size / 64) * MUTATION_RATE;

    for (int32_t i = 0; i < num_mutations; i++) {
        int32_t bit_index = xorshift() % model->param->size;
        bitset_toggle(model->param, bit_index);
    }
}

void nnmodel_train(struct nnmodel* model) {
    const int32_t max_iterations = 10000000;
    char input_buffer[256];
    int32_t teacher_length = strlen(model->teacher);

    input_buffer[0] = model->teacher[0];
    input_buffer[1] = '\0';

    nnmodel_backup_save(model);

    xorshift_state = (uint32_t)time(NULL);

    int explore = 0;
    for (int32_t iteration = 0; iteration < max_iterations; iteration++) {
        int score = 0;
        int correct_predictions = 0;
        char ch_last = 0;

        nnmodel_calc(model, input_buffer, teacher_length);
        for (int i = 1; i < teacher_length; i++) {
            char ch_result = model->output[i];
            char ch_teacher = model->teacher[i];
            if (ch_result == ch_teacher) {
                score += 100;
                correct_predictions += 1;
            } else {
                if (ch_result != ch_last) {
                    score += 6;
                }
            }
            ch_last = ch_result;
        }

        if (score > model->bestscore) {
            model->bestscore = score;
            nnmodel_backup_save(model);
            printf("Iteration %d: New best score: %d (Accuracy: %d%%)\n",
                   iteration,
                   score,
                   (correct_predictions * 100) / (teacher_length - 1));

            if (correct_predictions == teacher_length - 1) {
                printf("Perfect prediction achieved! Training complete.\n");
                break;
            }
        } else {
            if (explore == EXPLORE_RATE) {
                nnmodel_backup_load(model);
                explore = 0;
            } else {
                explore += 1;
                nnmodel_mutation(model);
            }
        }
    }
}

struct nnmodel nnmodel_init(struct bitset* param, struct bitset* backup, struct bitset* buf1, struct bitset* buf2, const char* teacher, char* output) {
    struct nnmodel model = {
        .param = param,
        .backup = backup,
        .state = buf1,
        .buf = buf2,
        .teacher = teacher,
        .output = output,
        .bestscore = 0};

    for (int32_t i = 0; i < param->size; i++) {
        bitset_set0(param, i);
    }

    return model;
}

int main() {
    static uint64_t param_data[PARAM_BITSIZE / 64];
    static uint64_t backup_data[PARAM_BITSIZE / 64];
    static uint64_t state_data[LAYER_BITSIZE / 64];
    static uint64_t buf_data[LAYER_BITSIZE / 64];

    struct bitset param_bitset = bitset_init(param_data, PARAM_BITSIZE);
    struct bitset backup_bitset = bitset_init(backup_data, PARAM_BITSIZE);
    struct bitset state_bitset = bitset_init(state_data, LAYER_BITSIZE);
    struct bitset buf_bitset = bitset_init(buf_data, LAYER_BITSIZE);

    static char teacher[BUFFER_BITSIZE];
    static char output[BUFFER_BITSIZE];

    file_read(teacher, "./teacher.txt");

    struct nnmodel model = nnmodel_init(&param_bitset, &backup_bitset, &state_bitset, &buf_bitset, teacher, output);

    nnmodel_train(&model);

    printf("Final best score: %d\n", model.bestscore);

    return 0;
}
