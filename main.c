#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PARAM_BITSIZE (512 * 512 * 2)
#define BUFFER_BITSIZE (256 * 1024)
#define LAYER_BITSIZE 512
#define LAYER_DEPTH 2

struct bitset {
    uint64_t* data;
    int32_t size;
};

struct nnmodel {
    struct bitset* param;
    struct bitset* backup;
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
    for (int32_t depth_i = 0; depth_i < LAYER_DEPTH - 1; depth_i++) {
        for (int32_t output_i = 0; output_i < LAYER_BITSIZE; output_i++) {
            int8_t sum = 0;
            for (int32_t input_i = 0; input_i < LAYER_BITSIZE; input_i++) {
                sum += bitset_get(model->state1, input_i) && bitset_get(model->param, param_i);
                param_i++;
                sum -= bitset_get(model->state1, input_i) && bitset_get(model->param, param_i);
                param_i++;
            }
            if (sum > maxchar_value && output_i < 256) {
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
    return maxchar_index;
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

// Save current parameters to backup
void nnmodel_backup_save(struct nnmodel* model) {
    memcpy(model->backup->data, model->param->data, model->param->size / 8);
}

// Load parameters from backup
void nnmodel_backup_load(struct nnmodel* model) {
    memcpy(model->param->data, model->backup->data, model->param->size / 8);
}

// XORShift random number generator
static uint32_t xorshift_state = 1;

uint32_t xorshift(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift_state = x;
    return x;
}

// Perform random mutations on the network parameters
void nnmodel_mutation(struct nnmodel* model) {
    int32_t num_mutations = (model->param->size / 64);  // Mutate ~1% of parameters

    for (int32_t i = 0; i < num_mutations; i++) {
        int32_t bit_index = xorshift() % model->param->size;
        bitset_toggle(model->param, bit_index);
    }
}

// Training function implementing reinforcement learning
void nnmodel_train(struct nnmodel* model) {
    const int32_t max_iterations = 10000000;
    char input_buffer[256] = {0};
    int32_t teacher_length = strlen(model->teacher);

    // Save initial parameters
    nnmodel_backup_save(model);

    // Initialize random seed
    xorshift_state = (uint32_t)time(NULL);

    for (int32_t iteration = 0; iteration < max_iterations; iteration++) {
        int32_t score = 0;
        int32_t correct_predictions = 0;

        // Test current parameters by predicting each character
        for (int32_t i = 0; i < teacher_length - 1; i++) {
            // Copy partial input string for prediction
            strncpy(input_buffer, model->teacher, i + 1);
            input_buffer[i + 1] = '\0';

            // Predict next character
            uint8_t predicted_char = nnmodel_calc(model, input_buffer);
            uint8_t actual_char = (uint8_t)model->teacher[i + 1];

            // Score the prediction
            if (predicted_char == actual_char) {
                correct_predictions++;
                score += 100;  // Bonus for exact match
            } else {
                // Partial credit for close predictions (ASCII difference)
                int32_t char_diff = abs(predicted_char - actual_char);
                if (char_diff < 10) {
                    score += (10 - char_diff) * 5;
                }
            }
        }

        // Calculate final score including accuracy percentage
        score = score + (correct_predictions * 100 / (teacher_length - 1));

        // If new score is better, save parameters
        if (score > model->bestscore) {
            model->bestscore = score;
            nnmodel_backup_save(model);
            printf("Iteration %d: New best score: %d (Accuracy: %d%%)\n",
                   iteration,
                   score,
                   (correct_predictions * 100) / (teacher_length - 1));

            // Early stopping if we achieve perfect prediction
            if (correct_predictions == teacher_length - 1) {
                printf("Perfect prediction achieved! Training complete.\n");
                break;
            }
        } else {
            // If score is worse, revert to backup
            nnmodel_backup_load(model);
            // Perform mutation for next iteration
            nnmodel_mutation(model);
        }
    }
}

struct nnmodel nnmodel_init(struct bitset* param, struct bitset* backup, struct bitset* buf1, struct bitset* buf2, const char* teacher) {
    struct nnmodel model = {
        .param = param,
        .backup = backup,
        .state1 = buf1,
        .state2 = buf2,
        .teacher = teacher,
        .bestscore = 0};

    // Initialize parameters randomly
    for (int32_t i = 0; i < param->size; i++) {
        if (xorshift() & 1) {
            bitset_set1(param, i);
        } else {
            bitset_set0(param, i);
        }
    }

    return model;
}

int main() {
    static uint64_t param_data[PARAM_BITSIZE / 64];
    static uint64_t backup_data[PARAM_BITSIZE / 64];
    static uint64_t buf1_data[BUFFER_BITSIZE / 64];
    static uint64_t buf2_data[BUFFER_BITSIZE / 64];

    struct bitset param_bitset = bitset_init(param_data, PARAM_BITSIZE);
    struct bitset backup_bitset = bitset_init(backup_data, PARAM_BITSIZE);
    struct bitset buf1_bitset = bitset_init(buf1_data, BUFFER_BITSIZE);
    struct bitset buf2_bitset = bitset_init(buf2_data, BUFFER_BITSIZE);

    static const char* foo_teacher = "this is a pen";
    struct nnmodel model = nnmodel_init(&param_bitset, &backup_bitset, &buf1_bitset, &buf2_bitset, foo_teacher);

    // Train the model
    nnmodel_train(&model);

    // Test the trained model
    printf("Final best score: %d\n", model.bestscore);

    return 0;
}