#include <iostream>
#include "net.h"
#include "modelbin.h"
#include "datareader.h"

using namespace ncnn;

int main(int argc, char** argv)
{
    // const char* bin_name = argc>1?argv[1]:"";
    // int num = argc>2? atoi(argv[2]): 10;
    // FILE* fp = fopen(bin_name, "rb");
    // if (!fp) {
    //     std::cout<<"open failed."<<std::endl;
    // }
    // DataReaderFromStdio dr(fp);
    // ModelBinFromDataReader mb(dr);
    // Mat weight_data = mb.load(num, 0);
    // for (int i=0; i<num; ++i){
    //     printf("%f \n", weight_data[i]);
    // }

    const char* param_file = argc > 1 ? argv[1] : "";
    const char* model_file = argc > 2 ? argv[2] : "";
    const char* out_name = argc > 3 ? argv[3] : "";
    Net net;
    net.load_param(param_file);
    net.load_model(model_file);
    int input_ids[8] = {101, 4937, 2003, 1037, 8403, 2611, 1012, 102};
    Mat in(8, input_ids), out;
    Extractor ex = net.create_extractor();
    ex.input("input", in);
    ex.extract(out_name, out);
    for (int i = 0; i < 8 * 768; ++i)
    {
        printf("%f \n", out[i]);
    }
    return 0;
}