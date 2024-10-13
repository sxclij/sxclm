#include <fcntl.h>
#include <unistd.h>

#define sxclm_param_size (1 << 16)
#define sxclm_traning_size (1 << 16)
#define sxclm_random_size (1 << 8)
#define sxclm_param_path "test/1/param.txt"
#define sxclm_traning_path "test/1/traning.txt"

typedef char i8;
typedef int i32;
typedef long long unsigned u64;

struct sxclm_vec {
    union {
        i8* i8;
        u64* u64;
    } data;
    i32 size;
};
struct sxclm_model {
    struct sxclm_vec param;
    struct sxclm_vec traning;
    struct sxclm_vec out;
    i32 bestscore;
};

u64 xorshift(u64 x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}
i32 file_read(const char* path, i8* dst, i32 size) {
    i32 fd = open(path, O_RDONLY);
    read(fd, dst, size);
    return close(fd);
}

void sxclm_load(struct sxclm_model* model, i8* param_data, i8* traning_data, i8* out_data) {
    *model = (struct sxclm_model){
        .param = (struct sxclm_vec){.data = param_data, sxclm_param_size},
        .traning = (struct sxclm_vec){.data = traning_data, sxclm_traning_size},
        .out = (struct sxclm_vec){.data = out_data, 0},
        .bestscore = 0,
    };
    file_read(sxclm_param_path, model->param.data.i8,sxclm_param_size);
    model->traning.size = file_read(sxclm_traning_path, model->traning.data.i8,sxclm_traning_size);
}
void sxclm_init(struct sxclm_model* model) {
    model->out.size = 0;
}
void sxclm_rand(struct sxclm_vec* param) {
    u64 x1 = 0;
    u64 x2 = 0;
    for (i32 i = 0; i < sxclm_random_size; i++) {
        x1 = xorshift(x1);
        x2 = xorshift(x2);
        param->data.i8[x1 % param->size] = x2;
    }
}
void sxclm_calc(struct sxclm_vec* param, struct sxclm_vec* out) {
}
i32 sxclm_scoring(struct sxclm_vec* traning, struct sxclm_vec* out) {
}
void sxclm_save(struct sxclm_vec* param) {
    i32 fd = open(sxclm_param_path, O_WRONLY);
    write(fd, param->data.i8, param->size);
    close(fd);
}
void sxclm_exec(struct sxclm_model* model) {
    sxclm_init(model);
    sxclm_rand(&model->param);
    sxclm_calc(&model->param, &model->out);
    i32 thisscore = sxclm_scoring(&model->traning, &model->out);
    if (thisscore >= model->bestscore) {
        sxclm_save(&model->param);
        model->bestscore = thisscore;
    }
}

i32 main() {
    static struct sxclm_model model;
    static i8 param_data[sxclm_param_size];
    static i8 traning_data[sxclm_traning_size];
    static i8 out_data[sxclm_traning_size];

    sxclm_load(&model, param_data, traning_data, out_data);
    while (1) {
        sxclm_exec(&model);
    }

    return 0;
}
