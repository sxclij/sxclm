#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum result {
    RESULT_OK,
    RESULT_ERR,
};

struct const_vec {
    const char* const data;
    const int32_t size;
};
struct vec {
    char* data;
    int32_t size;
};

int main() {
    return 0;
}
