#include <fcntl.h>
#include <unistd.h>

#define sxclm_param_size (1 << 16)
#define sxclm_traning_size (1 << 16)
#define sxclm_random_size (1 << 8)
#define sxclm_param_path "test/1/param.txt"
#define sxclm_traning_path "test/1/traning.txt"

struct sxclm_vec {
    union {
        char* i8;
        long long unsigned* u64;
    } data;
    int size;
};
struct sxclm_model {
    struct sxclm_vec param;
    struct sxclm_vec traning;
    struct sxclm_vec out;
    int bestscore;
};

long long unsigned xorshift(long long unsigned x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

void sxclm_load(struct sxclm_model* model, char* param_data, char* traning_data, char* out_data) {
    *model = (struct sxclm_model){
        .param = (struct sxclm_vec){.data = param_data, sxclm_param_size},
        .traning = (struct sxclm_vec){.data = traning_data, sxclm_traning_size},
        .out = (struct sxclm_vec){.data = out_data, 0},
        .bestscore = 0,
    };
    int fd1 = open(sxclm_param_path, O_RDONLY);
    read(fd1, model->param.data.i8, sxclm_traning_size);
    close(fd1);
    int fd2 = open(sxclm_traning_path, O_RDONLY);
    model->traning.size = read(fd2, model->traning.data.i8, sxclm_traning_size);
    close(fd2);
}
void sxclm_init(struct sxclm_model* model) {
    model->out.size = 0;
}
void sxclm_rand(struct sxclm_vec* param) {
    long long unsigned x1 = 0;
    long long unsigned x2 = 0;
    for (int i = 0; i < sxclm_random_size; i++) {
        x1 = xorshift(x1);
        x2 = xorshift(x2);
        param->data.i8[x1 % param->size] = x2;
    }
}
void sxclm_calc(struct sxclm_vec* param, struct sxclm_vec* out) {
}
int sxclm_scoring(struct sxclm_vec* traning, struct sxclm_vec* out) {
}
void sxclm_save(struct sxclm_vec* param) {
    int fd = open(sxclm_param_path, O_WRONLY);
    write(fd, param->data.i8, param->size);
    close(fd);
}
void sxclm_exec(struct sxclm_model* model) {
    sxclm_init(model);
    sxclm_rand(&model->param);
    sxclm_calc(&model->param, &model->out);
    int thisscore = sxclm_scoring(&model->traning, &model->out);
    if (thisscore >= model->bestscore) {
        sxclm_save(&model->param);
    }
}

int main() {
    static struct sxclm_model model;
    static char param_data[sxclm_param_size];
    static char traning_data[sxclm_traning_size];
    static char out_data[sxclm_traning_size];

    sxclm_load(&model, param_data, traning_data, out_data);
    while (1) {
        sxclm_exec(&model);
    }

    return 0;
}
