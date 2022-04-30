#include <stdio.h>

#include "cifa10.hpp"

static uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

std::vector<std::pair<int, cv::Mat>> result(60000);
int result_index = 0;

void load_cifa10_bacth(const std::string& cifa10_img_path, size_t batch_size) {
    //以二进制格式读取cifa10数据库中的图像文件和标签文件  
    std::ifstream cifa10_image(cifa10_img_path, std::ios::in | std::ios::binary);
    if (cifa10_image.is_open() == false)
    {
        cifa10_image.close();
        std::cout << "open cifa10 image file error!" << std::endl;
        return;
    }

    int rows = 32, cols = 32, ch = 3;
    
    for (int i = 0; i < batch_size; i++) {
        char label;
        cifa10_image.read(&label, 1);
        char* pixels = new char[rows * cols * ch];
        cifa10_image.read(pixels, rows * cols * ch);
        cv::Mat image(rows, cols, CV_8UC3);
        for (int c = 0; c != ch; c++) {
            for (int m = 0; m != rows; m++) {
                for (int n = 0; n != cols; n++) {
                    image.ptr<uchar>(m, n)[ch - c - 1] = pixels[c * rows * cols + m * rows + n];
                }
            }
        }
        result[result_index++] = (std::pair<int, cv::Mat>((int)label, std::move(image)));
    }
    cifa10_image.close();
}

std::vector<std::pair<int, cv::Mat>> load_cifa10(const std::string& cifa10_img_dir) {
    for (int i = 1; i <= 5; i++) {
        char file[512];
        sprintf(file, "%s\\data_batch_%d.bin", cifa10_img_dir.c_str(), i);
        load_cifa10_bacth(file, 10000);
    }
    std::vector<std::pair<int, cv::Mat>> ret = std::move(result);
    return ret;
}
