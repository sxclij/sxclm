#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// ---------- Configuration macros -------------
#define THREAD_COUNT 12
#define BUFFER_BITSIZE (1024 * 8)
#define OUTPUT_BITSIZE 1024  // in bytes for output string
#define LAYER_BITSIZE 1024
#define LAYER_DEPTH 4
#define PARAM_BITSIZE (LAYER_BITSIZE * LAYER_BITSIZE * (LAYER_DEPTH - 1) * 2)
#define EXPLORE_RATE 72
#define COMPLETE_RATE 0.85
#define MUTATION_RATE 0.0004

// ---------- Data structures -------------
struct vec {
    char* data;  // using a statically allocated char buffer
    int32_t size;
};

struct bitset {
    uint64_t* data;  // pointer to pre-allocated data array
    int32_t size;    // number of bits
};

// ---------- Global variables (statically allocated) -------------
// Teacher text is loaded from file into this global buffer.
#define TEACHER_BUFFER_SIZE (1024 * 16)
static char teacher_buffer[TEACHER_BUFFER_SIZE];
static const char* teacher = NULL;

// Global backup candidate for the parameters (best candidate so far)
static struct bitset backup;
static uint64_t backup_data[PARAM_BITSIZE / 64];
static int32_t bestscore = 0;

// Per-thread candidate data (indexed by thread id)
static uint64_t param_data[THREAD_COUNT][PARAM_BITSIZE / 64];
static uint64_t state_data[THREAD_COUNT][LAYER_BITSIZE / 64];
static uint64_t buf_data[THREAD_COUNT][LAYER_BITSIZE / 64];
static char output_data[THREAD_COUNT][OUTPUT_BITSIZE];

int tid_data[THREAD_COUNT];
pthread_t threads[THREAD_COUNT];
static uint32_t thread_rand[THREAD_COUNT];
static struct bitset param[THREAD_COUNT];
static struct bitset state[THREAD_COUNT];
static struct bitset buf[THREAD_COUNT];
static struct vec output[THREAD_COUNT];

// Flag to signal that training is complete (e.g. perfect prediction)
static volatile int training_done = 0;

// Mutex to protect updates to bestscore and backup candidate
static pthread_mutex_t bestscore_mutex = PTHREAD_MUTEX_INITIALIZER;

// ---------- Utility functions -------------
// Vector (string) functions.
void vec_init(struct vec* v, char* data) {
    v->data = data;
    v->size = 0;
    v->data[0] = '\0';
}

void vec_clear(struct vec* v) {
    v->size = 0;
    v->data[0] = '\0';
}

void vec_push_back(struct vec* v, char c) {
    // Overflow handling is removed per request.
    v->data[v->size++] = c;
    v->data[v->size] = '\0';
}

// File-reading utility (reads a file into a provided buffer).
void file_read(char* dst, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror(filename);
        return;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    if (file_size >= TEACHER_BUFFER_SIZE)
        file_size = TEACHER_BUFFER_SIZE - 1;
    fread(dst, 1, file_size, file);
    fclose(file);
    dst[file_size] = '\0';
}

// Write backup candidate to "nn.bin"
void write_backup_to_file(const void* data, size_t bytes) {
    FILE* file = fopen("nn.bin", "wb");
    if (!file) {
        perror("nn.bin");
        return;
    }
    fwrite(data, 1, bytes, file);
    fclose(file);
}

// Load backup candidate from "nn.bin" into backup.data.
void load_backup_from_file(void* data, size_t bytes) {
    FILE* file = fopen("nn.bin", "rb");
    if (file) {
        fread(data, 1, bytes, file);
        fclose(file);
    }
}

// ---------- Bitset functions -------------
struct bitset bitset_init(uint64_t* data, int32_t size) {
    return (struct bitset){.data = data, .size = size};
}

void bitset_set0(struct bitset* bs, int32_t index) {
    bs->data[index / 64] &= ~(1ULL << (index % 64));
}

void bitset_set1(struct bitset* bs, int32_t index) {
    bs->data[index / 64] |= (1ULL << (index % 64));
}

void bitset_toggle(struct bitset* bs, int32_t index) {
    bs->data[index / 64] ^= (1ULL << (index % 64));
}

int bitset_get(const struct bitset* bs, int32_t index) {
    return (bs->data[index / 64] & (1ULL << (index % 64))) != 0;
}

void bitset_clear(struct bitset* bs) {
    memset(bs->data, 0, (bs->size + 7) / 8);
}

void bitset_swap(struct bitset* a, struct bitset* b) {
    struct bitset t = *a;
    *a = *b;
    *b = t;
}

// ---------- Dot product computation functions -------------
#ifdef __AVX512VPOPCNTDQ__
static int compute_dot(const uint64_t* state_ptr, const uint64_t* param_ptr, int start_bit) {
    int words_per_block = LAYER_BITSIZE / 64;
    const uint64_t* param_pos = param_ptr + (start_bit / 64);
    const uint64_t* param_neg = param_ptr + (start_bit / 64) + words_per_block;
    __m512i dot_sum = _mm512_setzero_si512();
    int blocks = words_per_block / 8;
    for (int j = 0; j < blocks; j++) {
        __m512i s = _mm512_loadu_si512((__m512i const*)(state_ptr + j * 8));
        __m512i p_pos = _mm512_loadu_si512((__m512i const*)(param_pos + j * 8));
        __m512i p_neg = _mm512_loadu_si512((__m512i const*)(param_neg + j * 8));
        __m512i masked_pos = _mm512_and_si512(s, p_pos);
        __m512i masked_neg = _mm512_and_si512(s, p_neg);
        __m512i cnt_pos = _mm512_popcnt_epi64(masked_pos);
        __m512i cnt_neg = _mm512_popcnt_epi64(masked_neg);
        __m512i diff = _mm512_sub_epi64(cnt_pos, cnt_neg);
        dot_sum = _mm512_add_epi64(dot_sum, diff);
    }
    uint64_t tmp[8];
    _mm512_storeu_si512(tmp, dot_sum);
    int dot = 0;
    for (int i = 0; i < 8; i++) {
        dot += tmp[i];
    }
    return dot;
}
#else
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

static int compute_dot(const uint64_t* state_ptr, const uint64_t* param_ptr, int start_bit) {
    int words_per_block = LAYER_BITSIZE / 64;
    const uint64_t* param_pos = param_ptr + (start_bit / 64);
    const uint64_t* param_neg = param_ptr + (start_bit / 64) + words_per_block;
    __m256i dot_sum = _mm256_setzero_si256();
    int blocks = words_per_block / 4;
    for (int j = 0; j < blocks; j++) {
        __m256i s = _mm256_loadu_si256((__m256i const*)(state_ptr + j * 4));
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
#endif

// ---------- Neural network calculation functions -------------
// Each function takes an extra parameter (tid) for the thread id.

void nn_calc_input(int tid, char ch) {
    bitset_clear(&state[tid]);
    bitset_set1(&state[tid], ch + 128);
}

void nn_calc_next(int tid) {
    int32_t param_i = 0;
    int8_t maxchar_index = 0;
    int8_t maxchar_value = -128;
    for (int32_t output_i = 0; output_i < LAYER_BITSIZE; output_i++) {
        int dot = compute_dot(state[tid].data, param[tid].data, param_i);
        param_i += 2 * LAYER_BITSIZE;
        if (dot > maxchar_value && output_i < 256) {
            maxchar_index = output_i - 128;
            maxchar_value = dot;
        }
        if (dot > 0) {
            bitset_set1(&buf[tid], output_i);
        } else {
            bitset_set0(&buf[tid], output_i);
        }
    }
    bitset_swap(&state[tid], &buf[tid]);
    vec_push_back(&output[tid], maxchar_index);
}

void nn_calc(int tid, const char* input, int32_t count) {
    int32_t input_size = (int32_t)strlen(input);
    vec_clear(&output[tid]);
    bitset_clear(&state[tid]);
    for (int32_t i = 0; i < input_size; i++) {
        nn_calc_input(tid, input[i]);
        nn_calc_next(tid);
    }
    for (int32_t i = input_size; i < input_size + count; i++) {
        nn_calc_next(tid);
    }
}

// ---------- Mutation and backup functions -------------
static inline uint32_t xorshift_r(uint32_t* state_ptr) {
    uint32_t x = *state_ptr;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state_ptr = x;
    return x;
}

void nn_mutation(int tid) {
    int32_t num_mutations = (param[tid].size / 64) * MUTATION_RATE;
    for (int32_t i = 0; i < num_mutations; i++) {
        int32_t bit_index = xorshift_r(&thread_rand[tid]) % param[tid].size;
        bitset_toggle(&param[tid], bit_index);
    }
}

// ---------- Training function run by each thread -------------
void* train_thread(void* arg) {
    int tid = *(int*)arg;

    char input_buffer[BUFFER_BITSIZE];
    int teacher_length = (int)strlen(teacher);
    int input_len = teacher_length / 2;
    memcpy(input_buffer, teacher, input_len);
    input_buffer[input_len] = '\0';

    thread_rand[tid] = (uint32_t)time(NULL) + tid;
    int explore = 0;
    int iteration = 0;
    while (!training_done) {
        int score = 0;
        int correct_predictions = 0;
        int correct_last = -1;
        vec_clear(&output[tid]);
        bitset_clear(&state[tid]);

        for (int i = 0; i < input_len; i++) {
            nn_calc_input(tid, input_buffer[i]);
            nn_calc_next(tid);
            char ch_result = output[tid].data[i];
            char ch_teacher = teacher[i];
            if (ch_result == ch_teacher) {
                score += 100;
                if (correct_last == i - 1) {
                    score += 10000;
                    correct_predictions++;
                }
                correct_last = i;
            }
        }

        if (score > bestscore) {
            pthread_mutex_lock(&bestscore_mutex);
            if (score > bestscore) {
                bestscore = score;
                memcpy(backup.data, param[tid].data, (param[tid].size + 7) / 8);
                write_backup_to_file(backup.data, (backup.size + 7) / 8);
                printf("Thread %d, Iteration %d: New best score: %d (Accuracy: %d%%)\n",
                       tid, iteration, score,
                       (correct_predictions * 100) / (input_len ? input_len : 1));
                if (correct_predictions == input_len * COMPLETE_RATE) {
                    printf("Thread %d: Perfect prediction achieved! Training complete.\n", tid);
                    training_done = 1;
                }
            }
            pthread_mutex_unlock(&bestscore_mutex);
        } else {
            if (explore == EXPLORE_RATE) {
                pthread_mutex_lock(&bestscore_mutex);
                memcpy(param[tid].data, backup.data, (backup.size + 7) / 8);
                pthread_mutex_unlock(&bestscore_mutex);
                explore = 0;
            } else {
                explore++;
                nn_mutation(tid);
            }
        }
        iteration++;
    }
    return NULL;
}

// ---------- The new generate() function -------------
// This function reads seed input from "./input.txt", runs the network (using thread 0’s candidate),
// generates extra characters, and writes the complete output to "./output.txt".
void generate() {
    char input_buffer[BUFFER_BITSIZE];
    // Read the seed text from "./input.txt"
    file_read(input_buffer, "./input.txt");
    if (strlen(input_buffer) == 0) {
        printf("No seed input found in ./input.txt\n");
        return;
    }
    printf("Generating output from seed: \"%s\"\n", input_buffer);

    // Number of extra characters to generate (you can adjust this value)
    int extra_chars = 256;

    // For generation we use thread 0’s data. Reset its output and state:
    vec_clear(&output[0]);
    bitset_clear(&state[0]);

    // It is a good idea to load the backup candidate (best parameters) into thread 0 in case it was changed.
    pthread_mutex_lock(&bestscore_mutex);
    memcpy(param[0].data, backup.data, (backup.size + 7) / 8);
    pthread_mutex_unlock(&bestscore_mutex);

    // Run the network: first feed the seed, then generate extra characters.
    nn_calc(0, input_buffer, extra_chars);

    // Write the generated output to "./output.txt"
    FILE* fout = fopen("./output.txt", "w");
    if (!fout) {
        perror("Failed to open ./output.txt for writing");
        return;
    }
    fprintf(fout, "%s", output[0].data);
    fclose(fout);
    printf("Generated output written to ./output.txt\n");
}

// ---------- Initialization functions -------------
void global_init() {
    // Load teacher text.
    file_read(teacher_buffer, "./teacher.txt");
    teacher = teacher_buffer;

    // Initialize the global backup candidate.
    backup = bitset_init(backup_data, PARAM_BITSIZE);
    bitset_clear(&backup);

    // Attempt to load backup from "nn.bin" into backup.data.
    load_backup_from_file(backup.data, (backup.size + 7) / 8);

    // For each thread, initialize candidate data.
    for (int i = 0; i < THREAD_COUNT; i++) {
        param[i] = bitset_init(param_data[i], PARAM_BITSIZE);
        state[i] = bitset_init(state_data[i], LAYER_BITSIZE);
        buf[i] = bitset_init(buf_data[i], LAYER_BITSIZE);
        vec_init(&output[i], output_data[i]);
        // Load backup candidate into each thread's parameter array.
        memcpy(param[i].data, backup.data, (backup.size + 7) / 8);
    }
    for (int i = 0; i < THREAD_COUNT; i++) {
        tid_data[i] = i;
        pthread_create(&threads[i], NULL, train_thread, &tid_data[i]);
    }
}

// ---------- Main function -------------
int main() {
    global_init();

    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Training complete. Best score = %d\n", bestscore);

    // Generate output based on seed text from "./input.txt" and save to "./output.txt"
    generate();

    return 0;
}
