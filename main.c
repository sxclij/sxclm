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
    struct sxclm_vec out;
    int bestscore;
};

void sxclm_rand(struct sxclm_vec* param) {
}
void sxclm_calc(struct sxclm_model* param, struct sxclm_vec* out) {
}
int sxclm_scoring(struct sxclm_vec* traning, struct sxclm_vec* out) {
}
void sxclm_save(struct sxclm_vec* param) {
}
void sxclm_exec(struct sxclm_model* model) {
    sxclm_rand(&model->param);
    sxclm_calc(&model->param, &model->out);
    int thisscore = sxclm_scoring(&model->traning, &model->out);
    if (thisscore >= model->bestscore) {
        sxclm_save(&model->param);
    }
}

int main() {
    static struct sxclm_model model;
    char src[sxclm_src_capacity];

    int fd = open(sxclm_traning_path, O_RDONLY);
    int src_n = read(fd, src, sizeof(src) - 1);
    src[src_n] = '\0';
    close(fd);

    while (1) {
        sxclm_exec(&model);
    }

    return 0;
}
