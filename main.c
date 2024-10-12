#include <fcntl.h>
#include <unistd.h>

#define sxclm_src_capacity (1<<16)
#define sxclm_path "test/1/traning.txt"

int main() {
    char src[sxclm_src_capacity];

    int fd = open(sxclm_path , O_RDONLY);
    int src_n = read(fd, src, sizeof(src) - 1);
    src[src_n] = '\0';
    close(fd);

    write(1, src, src_n);
    write(1,"\n", 1);

    return 0;
}
