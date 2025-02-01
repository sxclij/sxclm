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
#define LEARNING_RATE 0.1f        // Learning rate for parameter updates
#define EPSILON 0.1f              // Exploration rate
#define GAMMA 0.99f              // Discount factor for future rewards

// Bitset data structure
typedef struct {
    uint64_t* bits;
    size_t size;
} Bitset;

// Experience replay buffer entry
typedef struct {
    char state[BUFFER_BYTE];
    int action;
    float reward;
    char next_state[BUFFER_BYTE];
    int done;
} Experience;

// Experience replay buffer
typedef struct {
    Experience* experiences;
    size_t capacity;
    size_t size;
    size_t position;
} ReplayBuffer;

// Neural network state
typedef struct {
    Bitset* params;
    ReplayBuffer* replay_buffer;
    float epsilon;
} NeuralNetwork;

// -------------------------
// Bitset functions
// -------------------------
Bitset* bitset_create(size_t size) {
    if (size == 0) return NULL;
    
    Bitset* bitset = malloc(sizeof(Bitset));
    if (!bitset) return NULL;
    
    size_t array_size = (size + 63) / 64;
    bitset->bits = calloc(array_size, sizeof(uint64_t));
    if (!bitset->bits) {
        free(bitset);
        return NULL;
    }
    
    bitset->size = size;
    return bitset;
}

void bitset_free(Bitset* bitset) {
    if (bitset) {
        free(bitset->bits);
        free(bitset);
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
    if (!bitset || index >= bitset->size) return 0;
    return (bitset->bits[index / 64] & (1ULL << (index % 64))) != 0;
}

// -------------------------
// Replay Buffer functions
// -------------------------
ReplayBuffer* replay_buffer_create(size_t capacity) {
    ReplayBuffer* buffer = malloc(sizeof(ReplayBuffer));
    if (!buffer) return NULL;
    
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

void replay_buffer_free(ReplayBuffer* buffer) {
    if (buffer) {
        free(buffer->experiences);
        free(buffer);
    }
}

void replay_buffer_add(ReplayBuffer* buffer, const char* state, int action, 
                      float reward, const char* next_state, int done) {
    if (!buffer) return;
    
    Experience* exp = &buffer->experiences[buffer->position];
    strncpy(exp->state, state, BUFFER_BYTE - 1);
    exp->state[BUFFER_BYTE - 1] = '\0';
    exp->action = action;
    exp->reward = reward;
    strncpy(exp->next_state, next_state, BUFFER_BYTE - 1);
    exp->next_state[BUFFER_BYTE - 1] = '\0';
    exp->done = done;
    
    buffer->position = (buffer->position + 1) % buffer->capacity;
    if (buffer->size < buffer->capacity) buffer->size++;
}

// -------------------------
// Neural Network functions
// -------------------------
int neural_calc(const Bitset* param, const char* source) {
    if (!param || !source) return -1;

    Bitset* input = bitset_create(LAYER_BITSIZE);
    Bitset* output = bitset_create(LAYER_BITSIZE);
    if (!input || !output) {
        bitset_free(input);
        bitset_free(output);
        return -1;
    }

    // Initialize input layer using the source string.
    size_t source_len = strlen(source);
    for (size_t i = 0; i < source_len && i < LAYER_BITSIZE/256; i++) {
        bitset_set1(input, 256 * i + (unsigned char)source[i]);
    }

    size_t param_i = 0;
    // Process through layers
    for (int depth_i = 0; depth_i < LAYER_DEPTH - 1; depth_i++) {
        int8_t output_max_value = INT8_MIN;
        int output_max_index = 0;

        // Calculate each output neuron
        for (int output_i = 0; output_i < LAYER_BITSIZE; output_i++) {
            int8_t input_sum = 0;

            // Process inputs for this neuron
            for (int input_i = 0; input_i < LAYER_BITSIZE; input_i++) {
                if (param_i + 1 >= param->size) {
                    bitset_free(input);
                    bitset_free(output);
                    return -1;
                }

                if (bitset_get(input, input_i)) {
                    input_sum += bitset_get(param, param_i) ? 1 : -1;
                }
                param_i += 2;
            }

            if (output_i < 128 && input_sum > output_max_value) {
                output_max_index = output_i;
                output_max_value = input_sum;
            }

            if (input_sum >= 1) {
                bitset_set1(output, output_i);
            } else {
                bitset_set0(output, output_i);
            }
        }

        if (depth_i == LAYER_DEPTH - 2) {
            int result = output_max_index;
            bitset_free(input);
            bitset_free(output);
            return result;
        }

        // Prepare for next layer: swap input and output
        Bitset* tmp = input;
        input = output;
        output = tmp;
    }

    bitset_free(input);
    bitset_free(output);
    return -1;
}

NeuralNetwork* neural_network_create() {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    if (!nn) return NULL;
    
    // Create parameter bitset (bytes converted to bits)
    nn->params = bitset_create(PARAM_BYTE * 8);
    nn->replay_buffer = replay_buffer_create(10000);  // Capacity for 10,000 experiences
    nn->epsilon = EPSILON;
    
    if (!nn->params || !nn->replay_buffer) {
        return NULL;
    }
    
    // Randomly initialize the parameters
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < nn->params->size; i++) {
        if (rand() % 2)
            bitset_set1(nn->params, i);
    }
    
    return nn;
}

int select_action(NeuralNetwork* nn, const char* state) {
    if ((float)rand() / RAND_MAX < nn->epsilon) {
        // Exploration: choose a random action (0 to 127)
        return rand() % 128;
    } else {
        // Exploitation: calculate action using the network
        return neural_calc(nn->params, state);
    }
}

void update_network(NeuralNetwork* nn) {
    if (!nn || nn->replay_buffer->size < 32) return;  // Minimum batch size

    const size_t batch_size = 32;
    for (size_t i = 0; i < batch_size; i++) {
        size_t idx = rand() % nn->replay_buffer->size;
        Experience* exp = &nn->replay_buffer->experiences[idx];
        
        // Calculate target Q-value
        float target = exp->reward;
        if (!exp->done) {
            int next_action = neural_calc(nn->params, exp->next_state);
            target += GAMMA * next_action;  // Simplified Q-learning update
        }
        
        // Calculate current Q-value
        int current = neural_calc(nn->params, exp->state);
        
        // Update parameters if error exceeds learning rate
        float error = target - current;
        if (fabs(error) > LEARNING_RATE) {
            size_t param_start = exp->action * LAYER_BITSIZE * 2;
            for (size_t j = 0; j < LAYER_BITSIZE * 2; j++) {
                if (rand() % 100 < fabs(error) * 100) {
                    bitset_toggle(nn->params, param_start + j);
                }
            }
        }
    }
    
    // Decay epsilon gradually
    nn->epsilon *= 0.995f;
    if (nn->epsilon < 0.01f)
        nn->epsilon = 0.01f;
}

int save_parameters(const NeuralNetwork* nn, const char* filename) {
    if (!nn || !nn->params || !filename) {
        return -1;
    }

    FILE* file = fopen(filename, "wb");
    if (!file) {
        return -1;
    }

    size_t param_size = nn->params->size;
    size_t array_size = (param_size + 63) / 64;
    
    if (fwrite(&param_size, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fwrite(&nn->epsilon, sizeof(float), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fwrite(nn->params->bits, sizeof(uint64_t), array_size, file) != array_size) {
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

NeuralNetwork* load_parameters(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }

    size_t param_size;
    if (fread(&param_size, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fclose(file);
        return NULL;
    }

    if (fread(&nn->epsilon, sizeof(float), 1, file) != 1) {
        free(nn);
        fclose(file);
        return NULL;
    }

    nn->params = bitset_create(param_size);
    if (!nn->params) {
        free(nn);
        fclose(file);
        return NULL;
    }

    size_t array_size = (param_size + 63) / 64;
    if (fread(nn->params->bits, sizeof(uint64_t), array_size, file) != array_size) {
        bitset_free(nn->params);
        free(nn);
        fclose(file);
        return NULL;
    }

    nn->replay_buffer = replay_buffer_create(10000);
    if (!nn->replay_buffer) {
        bitset_free(nn->params);
        free(nn);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return nn;
}

// -------------------------
// Utility Functions
// -------------------------

// Reads the entire contents of a file into a dynamically allocated string.
// The caller is responsible for freeing the returned string.
char* read_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file)
        return NULL;
    
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    rewind(file);
    
    char* buffer = malloc(length + 1);
    if (buffer) {
        fread(buffer, 1, length, file);
        buffer[length] = '\0';
    }
    fclose(file);
    return buffer;
}

// -------------------------
// Main function
// -------------------------
int main() {
    // Print teacher instructions (if available) from teacher.txt
    char* teacherText = read_file("teacher.txt");
    if (teacherText) {
        printf("=== Teacher Instructions ===\n%s\n", teacherText);
        free(teacherText);
    } else {
        printf("teacher.txt not found.\n");
    }
    
    // Read the initial state from input.txt
    char init_state[BUFFER_BYTE] = "Initial state"; // default value
    FILE* fin = fopen("input.txt", "r");
    if (fin) {
        if (fgets(init_state, BUFFER_BYTE, fin) != NULL) {
            // Remove trailing newline if any
            size_t len = strlen(init_state);
            if (len > 0 && init_state[len - 1] == '\n')
                init_state[len - 1] = '\0';
        }
        fclose(fin);
    } else {
        printf("input.txt not found. Using default initial state.\n");
    }
    
    // Open output.txt for writing results.
    FILE* fout = fopen("output.txt", "w");
    if (!fout) {
        printf("Failed to open output.txt for writing.\n");
        return 1;
    }
    
    // Create and initialize the neural network.
    NeuralNetwork* nn = neural_network_create();
    if (!nn) {
        fprintf(fout, "Failed to create neural network\n");
        fclose(fout);
        return 1;
    }
    
    // Training loop: run 100 episodes using the initial state from input.txt.
    for (int episode = 0; episode < 100; episode++) {
        char state[BUFFER_BYTE];
        strncpy(state, init_state, BUFFER_BYTE - 1);
        state[BUFFER_BYTE - 1] = '\0';
        float total_reward = 0;
        int done = 0;
        
        // Run one episode until termination condition is met.
        while (!done) {
            int action = select_action(nn, state);
            float reward = (float)(action % 10) / 10.0f;
            char next_state[BUFFER_BYTE];
            snprintf(next_state, BUFFER_BYTE, "State after action %d", action);
            done = (reward > 0.8f);
            
            replay_buffer_add(nn->replay_buffer, state, action, reward, next_state, done);
            update_network(nn);
            
            strncpy(state, next_state, BUFFER_BYTE - 1);
            state[BUFFER_BYTE - 1] = '\0';
            total_reward += reward;
        }
        
        // Write episode results to output.txt
        fprintf(fout, "Episode %d: Total reward = %.2f\n", episode, total_reward);
        
        // Save parameters every 10 episodes
        if (episode % 10 == 0) {
            if (save_parameters(nn, "neural_network.bin") == 0) {
                fprintf(fout, "Parameters saved successfully\n");
            } else {
                fprintf(fout, "Failed to save parameters\n");
            }
        }
    }
    
    // Demonstrate loading parameters from file.
    NeuralNetwork* loaded_nn = load_parameters("neural_network.bin");
    if (loaded_nn) {
        fprintf(fout, "Successfully loaded parameters\n");
        // Clean up loaded network.
        bitset_free(loaded_nn->params);
        replay_buffer_free(loaded_nn->replay_buffer);
        free(loaded_nn);
    }
    
    // Clean up original network.
    bitset_free(nn->params);
    replay_buffer_free(nn->replay_buffer);
    free(nn);
    
    fclose(fout);
    printf("Training complete. Check output.txt for results.\n");
    
    return 0;
}
