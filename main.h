/**
 * @file neural_network.h
 * @brief High-performance neural network implementation with bitset-based parameters
 * 
 * This header defines the public API for a binary neural network implementation
 * optimized for performance and memory efficiency. The network uses bitset-based
 * parameter storage and supports experience replay for reinforcement learning.
 *
 * Thread-safe operations are supported through internal mutex protection.
 *
 * @version 1.0.0
 * @author Improved Implementation
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Version information constants
 */
#define NN_VERSION_MAJOR 1
#define NN_VERSION_MINOR 0
#define NN_VERSION_PATCH 0

/**
 * @brief Maximum buffer size for state representations
 */
#define NN_MAX_STATE_SIZE 1024

/**
 * @brief Error codes returned by neural network operations
 */
typedef enum {
    NN_SUCCESS = 0,                  /**< Operation completed successfully */
    NN_ERROR_MEMORY = -1,           /**< Memory allocation failed */
    NN_ERROR_INVALID_PARAM = -2,    /**< Invalid parameter provided */
    NN_ERROR_FILE_IO = -3,          /**< File I/O operation failed */
    NN_ERROR_BUFFER_FULL = -4,      /**< Internal buffer is full */
    NN_ERROR_INSUFFICIENT_DATA = -5, /**< Not enough data for operation */
    NN_ERROR_NOT_INITIALIZED = -6,   /**< Neural network not initialized */
    NN_ERROR_THREAD_ERROR = -7      /**< Thread operation failed */
} NNError;

/**
 * @brief Network configuration parameters
 */
typedef struct {
    size_t param_byte_size;    /**< Size of parameter space in bytes */
    size_t buffer_size;        /**< Size of experience replay buffer */
    size_t layer_bitsize;      /**< Size of each layer in bits */
    size_t layer_depth;        /**< Number of layers in the network */
    float learning_rate;       /**< Learning rate for parameter updates */
    float initial_epsilon;     /**< Initial exploration rate */
    float gamma;              /**< Discount factor for future rewards */
    size_t batch_size;        /**< Training batch size */
    float epsilon_decay;      /**< Rate of exploration decay */
    float min_epsilon;        /**< Minimum exploration rate */
} NNConfig;

/**
 * @brief Network statistics information
 */
typedef struct {
    double average_reward;     /**< Average reward per episode */
    double average_loss;       /**< Average loss per training batch */
    size_t episodes_completed; /**< Total number of completed episodes */
    float current_epsilon;     /**< Current exploration rate */
    size_t total_steps;       /**< Total number of training steps */
} NNStats;

/**
 * @brief Opaque pointer to neural network instance
 */
typedef struct NeuralNetwork NeuralNetwork;

/**
 * @brief Creates a new neural network instance
 *
 * @param config Pointer to configuration structure
 * @param[out] nn Pointer to store created neural network
 * @return NNError code indicating success or failure
 */
NNError nn_create(const NNConfig* config, NeuralNetwork** nn);

/**
 * @brief Frees all resources associated with a neural network
 *
 * @param nn Neural network to free
 * @return NNError code indicating success or failure
 */
NNError nn_free(NeuralNetwork* nn);

/**
 * @brief Predicts action for given state
 *
 * @param nn Neural network instance
 * @param state Current state string
 * @param[out] action Selected action
 * @return NNError code indicating success or failure
 */
NNError nn_predict(const NeuralNetwork* nn, const char* state, int* action);

/**
 * @brief Trains network on a single experience
 *
 * @param nn Neural network instance
 * @param state Current state string
 * @param action Taken action
 * @param reward Received reward
 * @param next_state Resulting state string
 * @param done Whether episode ended
 * @return NNError code indicating success or failure
 */
NNError nn_train(NeuralNetwork* nn, const char* state, int action, 
                 float reward, const char* next_state, int done);

/**
 * @brief Saves network state to file
 *
 * @param nn Neural network instance
 * @param filename Path to save file
 * @return NNError code indicating success or failure
 */
NNError nn_save(const NeuralNetwork* nn, const char* filename);

/**
 * @brief Loads network state from file
 *
 * @param filename Path to load file
 * @param[out] nn Pointer to store loaded neural network
 * @return NNError code indicating success or failure
 */
NNError nn_load(const char* filename, NeuralNetwork** nn);

/**
 * @brief Retrieves current network statistics
 *
 * @param nn Neural network instance
 * @param[out] stats Pointer to statistics structure
 * @return NNError code indicating success or failure
 */
NNError nn_get_statistics(const NeuralNetwork* nn, NNStats* stats);

/**
 * @brief Returns string description of error code
 *
 * @param error Error code to describe
 * @return Constant string describing the error
 */
const char* nn_error_string(NNError error);

/**
 * @brief Retrieves library version information
 *
 * @param[out] major Major version number
 * @param[out] minor Minor version number
 * @param[out] patch Patch version number
 */
void nn_get_version(int* major, int* minor, int* patch);

/**
 * @brief Configuration validation function
 *
 * @param config Configuration to validate
 * @return NNError code indicating validity
 */
NNError nn_validate_config(const NNConfig* config);

/**
 * @brief Returns default configuration
 *
 * @return NNConfig structure with default values
 */
NNConfig nn_default_config(void);

/**
 * @brief Resets network to initial state
 *
 * @param nn Neural network instance
 * @return NNError code indicating success or failure
 */
NNError nn_reset(NeuralNetwork* nn);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_H