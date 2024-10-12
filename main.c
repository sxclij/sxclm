#include <fcntl.h>
#include <unistd.h>

#define sxclm_src_capacity (1 << 16)
#define sxclm_traning_path "test/1/traning.txt"
#define sxclm_save_path "test/1/save.txt"

struct sxclm_vec {
    char* data;
    int size;
};

struct sxclm_model {
    struct sxclm_vec traning;
    struct sxclm_vec param;
    int bestscore;
};

void sxclm_rand() {
}
void sxclm_calc() {
}
void sxclm_cost() {
}
void sxclm_save() {
}
void sxclm_exec() {
    sxclm_rand();
    sxclm_calc();
    sxclm_cost();
    if (1) {
        sxclm_save();
    }
}

int main() {
    char src[sxclm_src_capacity];

    int fd = open(sxclm_traning_path, O_RDONLY);
    int src_n = read(fd, src, sizeof(src) - 1);
    src[src_n] = '\0';
    close(fd);

    write(1, src, src_n);
    write(1, "\n", 1);

    while (1) {
        sxclm_exec();
    }

    return 0;
}
