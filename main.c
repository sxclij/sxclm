#include <fcntl.h>
#include <unistd.h>

#define sxclm_param_size (1 << 16)
#define sxclm_traning_size (1 << 16)
#define sxclm_param_path "test/1/param.txt"
#define sxclm_traning_path "test/1/traning.txt"

struct sxclm_vec {
    char* data;
    int size;
};
struct sxclm_model {
    struct sxclm_vec param;
    struct sxclm_vec traning;
    struct sxclm_vec out;
    int bestscore;
};

void sxclm_load(struct sxclm_model* model, char* param_data, char* traning_data, char* out_data) {
    int fd;
    *model = (struct sxclm_model) {
        .param = (struct sxclm_vec) {.data = param_data, sxclm_param_size},
        .traning = (struct sxclm_vec) {.data = traning_data, sxclm_traning_size},
        .out = (struct sxclm_vec) {.data = out_data, 0},
        .bestscore = 0,
    };
    fd = open(sxclm_param_path, O_RDONLY);
    model->param.size = read(fd, model->param.data, sxclm_traning_size);
    close(fd);
    fd = open(sxclm_traning_path, O_RDONLY);
    model->traning.size = read(fd, model->traning.data, sxclm_traning_size);
    close(fd);
}
void sxclm_init(struct sxclm_model* model) {
}
void sxclm_rand(struct sxclm_vec* param) {
}
void sxclm_calc(struct sxclm_vec* param, struct sxclm_vec* out) {
}
int sxclm_scoring(struct sxclm_vec* traning, struct sxclm_vec* out) {
}
void sxclm_save(struct sxclm_vec* param) {
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

    sxclm_load(&model,param_data,traning_data, out_data);
    while (1) {
        sxclm_exec(&model);
    }

    return 0;
}
