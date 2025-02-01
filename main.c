#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <limits.h>

// Configuration constants
#define PARAM_BYTE (1024 * 1024)  // 1MB parameter size (in bytes)
#define BUFFER_BYTE 1024          // Buffer size for reading files and states
#define LAYER_BITSIZE 512         // Size of each layer (in bits)
#define LAYER_DEPTH 3             // Number of layers
#define EPSILON 0.1f              // Exploration rate

// -------------------------
// Data Structures
// -------------------------

// Bitset data structure: stores bits as an array of 64-bit integers.
typedef struct {
    uint64_t* bits;
    size_t size;  // total number of bits
} Bitset;

// Experience replay buffer entry (not used in inference but kept for completeness)
typedef struct {
    char state[BUFFER_BYTE];
    int action;
    float reward;
    char next_state[BUFFER_BYTE];
    int done;
} Experience;

// Replay buffer for experience replay (not used in inference but kept for completeness)
typedef struct {
    Experience* experiences;
    size_t capacity;
    size_t size;
    size_t position;
} ReplayBuffer;

// Neural network structure including parameters and replay buffer.
typedef struct {
    Bitset* params;
    ReplayBuffer* replay_buffer;
    float epsilon;
} NeuralNetwork;

// -------------------------
// Bitset Functions
// -------------------------

// Create a bitset with the given number of bits.
Bitset* bitset_create(size_t size) {
    if (size == 0)
        return NULL;

    Bitset* bitset = malloc(sizeof(Bitset));
    if (!bitset)
        return NULL;

    size_t array_size = (size + 63) / 64;
    bitset->bits = calloc(array_size, sizeof(uint64_t));
    if (!bitset->bits) {
        free(bitset);
        return NULL;
    }

    bitset->size = size;
    return bitset;
}

// Free the memory allocated for a bitset.
void bitset_free(Bitset* bitset) {
    if (bitset) {
        free(bitset->bits);
        free(bitset);
    }
}

// Reset (clear) all bits in the bitset to 0.
void bitset_clear(Bitset* bitset) {
    if (bitset && bitset->bits) {
        size_t array_size = (bitset->size + 63) / 64;
        memset(bitset->bits, 0, array_size * sizeof(uint64_t));
    }
}

void bitset_set0(Bitset* bitset, size_t index) {
    if (bitset && index < bitset->size) {
        bitset->bits[index / 64] &= ~(1ULL << (index % 64));
    }
}

void bitset_set1(Bitset* bitset, size_t index) {
    if (bitset && index < bitset->size) {
        bitset->bits[index / 64] |= (1ULL << (index % 64));
    }
}

void bitset_toggle(Bitset* bitset, size_t index) {
    if (bitset && index < bitset->size) {
        bitset->bits[index / 64] ^= (1ULL << (index % 64));
    }
}

int bitset_get(const Bitset* bitset, size_t index) {
    if (!bitset || index >= bitset->size)
        return 0;
    return (bitset->bits[index / 64] & (1ULL << (index % 64))) != 0;
}

// -------------------------
// Replay Buffer Functions (Unused for Inference)
// -------------------------

// Create a replay buffer with the specified capacity.
ReplayBuffer* replay_buffer_create(size_t capacity) {
    ReplayBuffer* buffer = malloc(sizeof(ReplayBuffer));
    if (!buffer)
        return NULL;

    buffer->experiences = malloc(capacity * sizeof(Experience));
    if (!buffer->experiences) {
        free(buffer);
        return NULL;
    }

    buffer->capacity = capacity;
    buffer->size = 0;
    buffer->position = 0;
    return buffer;
}

// Free the memory allocated for a replay buffer.
void replay_buffer_free(ReplayBuffer* buffer) {
    if (buffer) {
        free(buffer->experiences);
        free(buffer);
    }
}

// Add an experience to the replay buffer.
void replay_buffer_add(ReplayBuffer* buffer, const char* state, int action,
                       float reward, const char* next_state, int done) {
    if (!buffer)
        return;

    Experience* exp = &buffer->experiences[buffer->position];
    strncpy(exp->state, state, BUFFER_BYTE - 1);
    exp->state[BUFFER_BYTE - 1] = '\0';
    exp->action = action;
    exp->reward = reward;
    strncpy(exp->next_state, next_state, BUFFER_BYTE - 1);
    exp->next_state[BUFFER_BYTE - 1] = '\0';
    exp->done = done;

    buffer->position = (buffer->position + 1) % buffer->capacity;
    if (buffer->size < buffer->capacity)
        buffer->size++;
}

// -------------------------
// Neural Network Functions
// -------------------------

// Calculate the network output given the parameter bitset and an input source string.
// This function simulates a multi-layer network where each layer processes bits.
int neural_calc(const Bitset* param, const char* source) {
    if (!param || !source)
        return -1;

    // Create input and output bitsets for the layers.
    Bitset* input = bitset_create(LAYER_BITSIZE);
    Bitset* output = bitset_create(LAYER_BITSIZE);
    if (!input || !output) {
        bitset_free(input);
        bitset_free(output);
        return -1;
    }

    // Initialize the input layer using the source string.
    // (Only a few bytes are used to set bits in the input layer.)
    size_t source_len = strlen(source);
    size_t max_bytes = LAYER_BITSIZE / 256; // how many characters to use
    for (size_t i = 0; i < source_len && i < max_bytes; i++) {
        // Use the ASCII code of the character to set a bit in a block of 256 bits.
        bitset_set1(input, 256 * i + (unsigned char)source[i]);
    }

    size_t param_i = 0;
    // Process the input through multiple layers.
    for (int depth_i = 0; depth_i < LAYER_DEPTH - 1; depth_i++) {
        int8_t output_max_value = INT8_MIN;
        int output_max_index = 0;

        // Clear the output bitset before using it.
        bitset_clear(output);

        // Process each output neuron.
        for (int output_i = 0; output_i < LAYER_BITSIZE; output_i++) {
            int8_t input_sum = 0;

            // Process each input neuron for the current output neuron.
            for (int input_i = 0; input_i < LAYER_BITSIZE; input_i++) {
                // Ensure that we do not run out of parameters.
                if (param_i + 1 >= param->size) {
                    bitset_free(input);
                    bitset_free(output);
                    return -1;
                }

                // If the input neuron is active, use two bits from the parameters
                // to decide whether to add or subtract.
                if (bitset_get(input, input_i)) {
                    input_sum += bitset_get(param, param_i) ? 1 : -1;
                }
                param_i += 2;
            }

            // For the first 128 output neurons, track the neuron with the highest sum.
            if (output_i < 128 && input_sum > output_max_value) {
                output_max_index = output_i;
                output_max_value = input_sum;
            }

            // Activate the output neuron if the sum exceeds a threshold.
            if (input_sum >= 1)
                bitset_set1(output, output_i);
            else
                bitset_set0(output, output_i);
        }

        // If this is the final layer, return the index of the neuron with the maximum value.
        if (depth_i == LAYER_DEPTH - 2) {
            int result = output_max_index;
            bitset_free(input);
            bitset_free(output);
            return result;
        }

        // Prepare for the next layer:
        // Swap the input and output bitsets. Before reusing the new output,
        // we will clear it at the beginning of the loop.
        Bitset* tmp = input;
        input = output;
        output = tmp;
    }

    bitset_free(input);
    bitset_free(output);
    return -1;
}

// Create and initialize a neural network.
NeuralNetwork* neural_network_create() {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    if (!nn)
        return NULL;

    nn->params = bitset_create(PARAM_BYTE * 8);
    nn->replay_buffer = replay_buffer_create(10000);  // Capacity for 10,000 experiences
    nn->epsilon = EPSILON;

    if (!nn->params || !nn->replay_buffer) {
        if (nn->params)
            bitset_free(nn->params);
        if (nn->replay_buffer)
            replay_buffer_free(nn->replay_buffer);
        free(nn);
        return NULL;
    }

    // Randomly initialize the parameters.
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < nn->params->size; i++) {
        if (rand() % 2)
            bitset_set1(nn->params, i);
    }

    return nn;
}

// -------------------------
// Utility Functions
// -------------------------

// Read the entire contents of a file into a dynamically allocated string.
// The caller is responsible for freeing the returned string.
char* read_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file)
        return NULL;

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long length = ftell(file);
    if (length < 0) {
        fclose(file);
        return NULL;
    }
    rewind(file);

    char* buffer = malloc(length + 1);
    if (buffer) {
        if (fread(buffer, 1, length, file) != (size_t)length) {
            free(buffer);
            fclose(file);
            return NULL;
        }
        buffer[length] = '\0';
    }
    fclose(file);
    return buffer;
}

// Print teacher instructions from "teacher.txt" if available.
void print_teacher_instructions(void) {
    char* teacherText = read_file("teacher.txt");
    if (teacherText) {
        printf("=== Teacher Instructions ===\n%s\n", teacherText);
        free(teacherText);
    } else {
        printf("teacher.txt not found.\n");
    }
}

// Read the initial state from "input.txt" or use a default value.
void read_initial_state(char* init_state, size_t size) {
    strncpy(init_state, "Initial state", size - 1);
    init_state[size - 1] = '\0';

    FILE* fin = fopen("input.txt", "r");
    if (fin) {
        if (fgets(init_state, size, fin) != NULL) {
            // Remove trailing newline if any.
            size_t len = strlen(init_state);
            if (len > 0 && init_state[len - 1] == '\n')
                init_state[len - 1] = '\0';
        }
        fclose(fin);
    } else {
        printf("input.txt not found. Using default initial state.\n");
    }
}

// Free the neural network and its allocated resources.
void cleanup_network(NeuralNetwork* nn) {
    if (nn) {
        if (nn->params)
            bitset_free(nn->params);
        if (nn->replay_buffer)
            replay_buffer_free(nn->replay_buffer);
        free(nn);
    }
}

// -------------------------
// Main Function (Inference Only)
// -------------------------

int main() {
    // Print teacher instructions if available.
    print_teacher_instructions();

    // Read the input state from "input.txt".
    char init_state[BUFFER_BYTE];
    read_initial_state(init_state, BUFFER_BYTE);

    // Create and initialize the neural network.
    NeuralNetwork* nn = neural_network_create();
    if (!nn) {
        printf("Failed to create neural network.\n");
        return 1;
    }

    // Perform inference: calculate the network's output for the input state.
    int inference_result = neural_calc(nn->params, init_state);

    // Open (or create) output.txt and write the inference result.
    FILE* fout = fopen("output.txt", "w");
    if (!fout) {
        printf("Failed to open output.txt for writing.\n");
        cleanup_network(nn);
        return 1;
    }

    fprintf(fout, "Inference result: %d\n", inference_result);
    fclose(fout);

    // Clean up allocated resources.
    cleanup_network(nn);

    printf("Inference complete. Check output.txt for results.\n");
    return 0;
}
