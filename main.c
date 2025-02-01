#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARAM_BYTE 1024 * 1024
#define LAYER_BYTE (512 / 8)
#define LAYER_DEPTH 3
#define BUFFER_BYTE 1024

typedef struct {
    uint64_t* bits;
    size_t size;
} Bitset;

Bitset* bitset_create(size_t size) {
    Bitset* bitset = malloc(sizeof(Bitset));
    if (bitset == NULL) {
        return NULL;
    }
    bitset->size = size;
    bitset->bits = calloc((size + 63) / 64, sizeof(uint64_t));
    if (bitset->bits == NULL) {
        free(bitset);
        return NULL;
    }
    return bitset;
}
void bitset_free(Bitset* bitset) {
    if (bitset) {
        free(bitset->bits);
        free(bitset);
    }
}
void bitset_set0(Bitset* bitset, size_t index) {
    bitset->bits[index / 64] &= ~(1UL << (index % 64));
}
void bitset_set1(Bitset* bitset, size_t index) {
    bitset->bits[index / 64] |= (1UL << (index % 64));
}
int bitset_get(Bitset* bitset, size_t index) {
    if (index >= bitset->size) {
        return 0;
    }
    return (bitset->bits[index / 64] & (1UL << (index % 64))) != 0;
}

char neural_calc(const char* source) {
    Bitset* input = bitset_create(LAYER_BYTE);
    Bitset* output = bitset_create(LAYER_BYTE);
    for (int i = 0; source[i] != '\0'; i++) {
        bitset_set1(input, 256 * i + source[i]);
    }
}