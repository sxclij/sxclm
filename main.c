#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define THREAD_COUNT 14
#define BUFFER_BITSIZE (1024 * 1024)
#define OUTPUT_BITSIZE (1024 * 512)
#define TEACHER_BITSIZE (1024 * 256)
#define TEACHER_PAIRS_MAX 256

#define LAYER_BITSIZE 1024
#define LAYER_DEPTH 6
#define PARAM_BITSIZE (LAYER_BITSIZE * LAYER_BITSIZE * (LAYER_DEPTH - 1) * 2)

#define COMPLETE_RATE 0.85
#define EXPLORE_RATE 12
#define MUTATION_RATE 0.0002
#define CANCELLATION_RATE 3

struct vec {
    char* data;
    int32_t size;
};

struct bitset {
    uint64_t* data;
    int32_t size;
};

struct teacher_pair {
    struct vec user;   // input seed
    struct vec sxclm;  // expected output
};

static char teacher_data[TEACHER_BITSIZE / 8];
struct teacher_pair teacher_pair[TEACHER_PAIRS_MAX];
static int teacher_pair_count = 0;
static int teacher_sxclm_totallen = 0;

static struct bitset backup;
static uint64_t backup_data[PARAM_BITSIZE / 64];
static int32_t bestscore = 0;

static uint64_t param_data[THREAD_COUNT][PARAM_BITSIZE / 64];
static uint64_t state_data[THREAD_COUNT][LAYER_BITSIZE / 64];
static uint64_t buf_data[THREAD_COUNT][LAYER_BITSIZE / 64];
static char input_data[THREAD_COUNT][OUTPUT_BITSIZE / 8];  // backing store for input vectors
static char output_data[THREAD_COUNT][OUTPUT_BITSIZE / 8];

int tid_data[THREAD_COUNT];
pthread_t threads[THREAD_COUNT];
static uint32_t thread_rand[THREAD_COUNT];
static struct bitset param[THREAD_COUNT];
static struct bitset state[THREAD_COUNT];
static struct bitset buf[THREAD_COUNT];
static struct vec output[THREAD_COUNT];
static struct vec input[THREAD_COUNT];  // now used for input instead of local buffers

static volatile int training_done = 0;

static pthread_mutex_t bestscore_mutex = PTHREAD_MUTEX_INITIALIZER;

void vec_init(struct vec* v, char* data) {
    v->data = data;
    v->size = 0;
    v->data[0] = '\0';
}

void vec_clear(struct vec* v) {
    v->size = 0;
    if (v->data)
        v->data[0] = '\0';
}

void vec_push_back(struct vec* v, char c) {
    v->data[v->size++] = c;
    v->data[v->size] = '\0';
}

void file_read(char* dst, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror(filename);
        return;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    fread(dst, 1, file_size, file);
    fclose(file);
    dst[file_size] = '\0';
}

void file_write(struct vec* src, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror(filename);
        return;
    }
    fwrite(src->data, 1, src->size, file);
    fclose(file);
}

void write_backup_to_file(const void* data, size_t bytes) {
    FILE* file = fopen("nn.bin", "wb");
    if (!file) {
        perror("nn.bin");
        return;
    }
    fwrite(data, 1, bytes, file);
    fclose(file);
}

void load_backup_from_file(void* data, size_t bytes) {
    FILE* file = fopen("nn.bin", "rb");
    if (file) {
        fread(data, 1, bytes, file);
        fclose(file);
    }
}

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

void bitset_cpy(struct bitset* dst, struct bitset* src) {
    memcpy(dst->data, src->data, (src->size + 7) / 8);
    dst->size = src->size;
}

void bitset_clear(struct bitset* bs) {
    memset(bs->data, 0, (bs->size + 7) / 8);
}

void bitset_swap(struct bitset* a, struct bitset* b) {
    struct bitset t = *a;
    *a = *b;
    *b = t;
}

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

// Feed a single character into the network by setting the corresponding bit.
void nn_calc_input(int tid, char ch) {
    memset(state[tid].data, 0, LAYER_BITSIZE / 8);
    bitset_set1(&state[tid], ch + 128);
}

// Process one “time‐step” of the network.
// This function computes one dot–product per output neuron, sets the corresponding bit in buf,
// and tracks the output “character” (for the first 256 outputs).
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

static inline uint32_t xorshift_r(int tid) {
    uint32_t x = thread_rand[tid];
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    thread_rand[tid] = x;
    return x;
}

void nn_mutation(int tid) {
    int32_t num_mutations = (param[tid].size / 64) * MUTATION_RATE;
    for (int32_t i = 0; i < num_mutations; i++) {
        int32_t bit_index = xorshift_r(tid) % param[tid].size;
        bitset_toggle(&param[tid], bit_index);
    }
}

/* ===== MODIFIED TRAINING THREAD =====
   Instead of using a single teacher_data string, this version randomly selects a teacher_pair
   and uses its "user" text as input and its "sxclm" text as the expected output.
   This allows each teacher pair to be both a training and a test example.
===================================================== */
void* train_thread(void* arg) {
    int tid = *(int*)arg;
    thread_rand[tid] = (uint32_t)time(NULL) + tid;

    int explore = 0;
    int iteration = 0;
    while (!training_done) {
        int score = 0;
        int correct_predictions = 0;
        int correct_last = -1;
        int endtest = 0;

        for (int i = 0; i < teacher_pair_count && !endtest; i++) {
            vec_clear(&output[tid]);
            bitset_clear(&state[tid]);
            for (int j = 0; j < teacher_pair[i].user.size; j++) {
                nn_calc_input(tid, teacher_pair[i].user.data[j]);
                nn_calc_next(tid);
            }
            for (int j = 0; j < teacher_pair[i].user.size; j++) {
                nn_calc_input(tid, '\0');
                nn_calc_next(tid);
                char ch_result = output[tid].data[i];
                char ch_teacher = teacher_pair[i].sxclm.data[j];
                if (ch_result == ch_teacher) {
                    score += 1;
                    // Reward sequential correct predictions.
                    if (correct_last == i - 1) {
                        score += 10000;
                        correct_predictions++;
                    }
                    correct_last = i;
                }
                // Early break if too many consecutive mistakes.
                if (i - correct_last == CANCELLATION_RATE) {
                    endtest = 1;
                    break;
                }
            }
        }

        // Update best score and backup parameters if improved.
        if (score > bestscore) {
            pthread_mutex_lock(&bestscore_mutex);
            if (score > bestscore) {
                bestscore = score;
                if (iteration != 0) {
                    memcpy(backup.data, param[tid].data, (backup.size + 7) / 8);
                    write_backup_to_file(backup.data, (backup.size + 7) / 8);
                    printf("Thread %d, Iteration %d: New best score: %d (Accuracy: %d%%)\n",
                           tid, iteration, score,
                           (correct_predictions * 100) / teacher_sxclm_totallen);
                    if (correct_predictions >= teacher_sxclm_totallen * COMPLETE_RATE) {
                        printf("Thread %d: Sufficient accuracy achieved! Training complete.\n", tid);
                        training_done = 1;
                    }
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

void generate() {
    // Use the first thread's input vector as the seed.
    vec_clear(&input[0]);
    file_read(input[0].data, "./input.txt");
    input[0].size = strlen(input[0].data);
    if (input[0].size == 0) {
        printf("No seed input found in ./input.txt\n");
        return;
    }
    printf("Generating output from seed: \"%s\"\n", input[0].data);

    bitset_cpy(&param[0], &backup);
    vec_clear(&output[0]);
    bitset_clear(&state[0]);

    // Process the seed.
    for (int32_t i = 0; i < input[0].size; i++) {
        nn_calc_input(0, input[0].data[i]);
        nn_calc_next(0);
    }
    // Generate additional output.
    for (int32_t i = input[0].size; i < input[0].size + OUTPUT_BITSIZE / 8; i++) {
        nn_calc_input(0, '\0');
        nn_calc_next(0);
    }

    file_write(&output[0], "./output.txt");

    printf("Generated output written to ./output.txt\n");
}

void global_init() {
    file_read(teacher_data, "./teacher.txt");

    backup = bitset_init(backup_data, PARAM_BITSIZE);
    bitset_clear(&backup);

    load_backup_from_file(backup.data, (backup.size + 7) / 8);

    for (int i = 0; i < THREAD_COUNT; i++) {
        param[i] = bitset_init(param_data[i], PARAM_BITSIZE);
        state[i] = bitset_init(state_data[i], LAYER_BITSIZE);
        buf[i] = bitset_init(buf_data[i], LAYER_BITSIZE);
        vec_init(&output[i], output_data[i]);
        vec_init(&input[i], input_data[i]);  // Initialize the input vector here
        memcpy(param[i].data, backup.data, (backup.size + 7) / 8);
    }
}

/* ===== TEACHER PARSER (USING struct vec) =====
   The teacher file is expected to contain a number of <user>…</user> blocks
   followed by <sxclm>…</sxclm> blocks. This parser extracts each pair into a
   teacher_pair struct.
===================================================== */
void parse_teacher_data(void) {
    char* pos = teacher_data;
    const char* user_open = "<user>";
    const char* user_close = "</user>";
    const char* sxclm_open = "<sxclm>";
    const char* sxclm_close = "</sxclm>";

    while (teacher_pair_count < TEACHER_PAIRS_MAX) {
        char* user_start = strstr(pos, user_open);
        if (!user_start)
            break;
        char* user_end = strstr(user_start, user_close);
        if (!user_end)
            break;
        user_end += strlen(user_close);  // include closing tag

        int user_len = (int)(user_end - user_start);

        // Find the sxclm block that follows.
        char* sxclm_start = strstr(user_end, sxclm_open);
        if (!sxclm_start)
            break;
        char* sxclm_end = strstr(sxclm_start, sxclm_close);
        if (!sxclm_end)
            break;
        sxclm_end += strlen(sxclm_close);

        int sxclm_len = (int)(sxclm_end - sxclm_start);
        teacher_sxclm_totallen += sxclm_len;

        // Store the parsed messages in the teacher_pairs array.
        teacher_pair[teacher_pair_count].user.data = user_start;
        teacher_pair[teacher_pair_count].user.size = user_len;
        teacher_pair[teacher_pair_count].sxclm.data = sxclm_start;
        teacher_pair[teacher_pair_count].sxclm.size = sxclm_len;

        teacher_pair_count++;
        pos = sxclm_end;
    }
}

int main() {
    global_init();

    /* --- Parse teacher data and print the parsed teacher pairs --- */
    parse_teacher_data();
    printf("Parsed teacher data (%d pairs):\n", teacher_pair_count);

    for (int i = 0; i < THREAD_COUNT; i++) {
        tid_data[i] = i;
        pthread_create(&threads[i], NULL, train_thread, &tid_data[i]);
    }
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Training complete. Best score = %d\n", bestscore);

    generate();

    // (Optional) Free allocated teacher pair strings here.
    for (int i = 0; i < teacher_pair_count; i++) {
        free(teacher_pair[i].user.data);
        free(teacher_pair[i].sxclm.data);
    }

    return 0;
}
