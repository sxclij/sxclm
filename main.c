#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PARAM_BITSIZE (1024 * 1024 * 128)
#define BUFFER_BITSIZE 256 * 1024
#define LAYER_BITSIZE 512
#define LAYER_DEPTH 3

struct bitset {
    uint64_t* data;
    int32_t size;
};
struct nnmodel {
    struct bitset* param;
    struct bitset* state1;
    struct bitset* state2;
    const char* teacher;
    int32_t bestscore;
};

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

uint8_t nnmodel_calc_ch(struct nnmodel* model, uint8_t ch) {
    for (int32_t i = 0; i < 256; i++) {
        bitset_set0(model->state1, i);
    }
    bitset_set1(model->state1, ch);
    int32_t param_i = 0;
    int8_t maxchar_index = 0;
    int8_t maxchar_value = 0;
    for (int32_t depth_i = 0; depth_i < LAYER_DEPTH; depth_i++) {
        for (int32_t output_i = 0; output_i < LAYER_BITSIZE; output_i++) {
            int8_t sum = 0;
            for (int32_t input_i = 0; input_i < LAYER_BITSIZE; input_i++) {
                sum += bitset_get(model->state1, input_i) && bitset_get(model->param, param_i);
                param_i++;
                sum -= bitset_get(model->state1, input_i) && bitset_get(model->param, param_i);
                param_i++;
            }
            if(sum > maxchar_value && output_i < 256) {
                maxchar_index = output_i;
                maxchar_value = sum;
            }
            if (sum > 0) {
                bitset_set1(model->state2, output_i);
            } else {
                bitset_set0(model->state2, output_i);
            }
        }
        bitset_swap(&model->state1, &model->state2);
    }
}
uint8_t nnmodel_calc(struct nnmodel* model, const char* input) {
    int32_t input_size = strlen(input);
    uint8_t ch;
    bitset_clear(model->state1);
    for (int32_t i = 0; i < input_size; i++) {
        ch = nnmodel_calc_ch(model, ((const uint8_t*)input)[i]);
    }
    return ch;
}
struct nnmodel nnmodel_init(struct bitset* param, struct bitset* buf1, struct bitset* buf2, const char* teacher) {
    return (struct nnmodel){.param = param, .state1 = buf1, .state2 = buf2, .teacher = teacher, .bestscore = 0};
}

int main() {
    static uint64_t param_data[PARAM_BITSIZE / 64];
    static uint64_t buf1_data[BUFFER_BITSIZE / 64];
    static uint64_t buf2_data[BUFFER_BITSIZE / 64];
    struct bitset param_bitset = bitset_init(param_data, PARAM_BITSIZE);
    struct bitset buf1_bitset = bitset_init(buf1_data, BUFFER_BITSIZE);
    struct bitset buf2_bitset = bitset_init(buf2_data, BUFFER_BITSIZE);
    static const char* foo_teacher = "this is a pen";
    struct nnmodel model = nnmodel_init(&param_bitset, &buf1_bitset, &buf2_bitset, foo_teacher);

    nnmodel_calc(&model, foo_teacher);

    return 0;
}