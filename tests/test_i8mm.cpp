#include "testutil.h"
#include "layer/arm/convolution_sgemm_int8.h"
int main(int argc, const char** argv)
{
    int w = 4, h = 4, c = 4;
    int8_t* data_a = (int8_t*)malloc(w * h * c * sizeof(int8_t));
    for (int i = 0; i < w * h * c; ++i) data_a[i] = (int8_t)(i + 1);
    ncnn::Mat a(w, h, c, data_a, 1, 1);
    ncnn::Mat kernel(2, 2, 4, data_a);
    ncnn::Mat output(2, 2);
    // printf("total: %d, %d\n", a.total(), kernel.total());
    // int8_t *p = (int8_t *)a.data, *pkernel = (int8_t*)kernel.data;
    // for (int i=0; i<a.total(); ++i) {
    //     printf("%d, ", p[i]);
    // }
    // printf("\n");
    // for (int i=0; i<kernel.total(); ++i) {
    //     printf("%d, ", pkernel[i]);
    // }
    // printf("\n");
    ncnn::Option opt;
    opt.num_threads = 1;
    convolution_im2col_sgemm_int8_neon(a, output, kernel, 2, 2, 1, 1, 2, 2, opt);
    return 0;
}