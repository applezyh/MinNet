#include "mnist.hpp"

static uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

std::vector<std::pair<int, cv::Mat>> load_mnist(const std::string& mnist_img_path, const std::string& mnist_label_path) {
    //以二进制格式读取mnist数据库中的图像文件和标签文件  
    std::ifstream mnist_image(mnist_img_path, std::ios::in | std::ios::binary);
    std::ifstream mnist_label(mnist_label_path, std::ios::in | std::ios::binary);
    if (mnist_image.is_open() == false)
    {
        mnist_image.close();
        mnist_label.close();
        std::cout << "open mnist image file error!" << std::endl;
        return std::vector<std::pair<int, cv::Mat>>();
    }
    if (mnist_label.is_open() == false)
    {
        mnist_image.close();
        mnist_label.close();
        std::cout << "open mnist label file error!" << std::endl;
        return std::vector<std::pair<int, cv::Mat>>();
    }

    uint32_t magic = 0;//文件中的魔术数(magic number)  
    uint32_t num_items = 0;//mnist图像集文件中的图像数目  
    uint32_t num_label = 0;//mnist标签集文件中的标签数目  
    uint32_t rows = 0;//图像的行数  
    uint32_t cols = 0;//图像的列数  
    //读魔术数  
    mnist_image.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051)
    {
        mnist_image.close();
        mnist_label.close();
        std::cout << "this is not the mnist image file" << std::endl;
        return std::vector<std::pair<int, cv::Mat>>();
    }
    mnist_label.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049)
    {
        mnist_image.close();
        mnist_label.close();
        std::cout << "this is not the mnist label file" << std::endl;
        return std::vector<std::pair<int, cv::Mat>>();
    }
    //读图像/标签数  
    mnist_image.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    mnist_label.read(reinterpret_cast<char*>(&num_label), 4);
    num_label = swap_endian(num_label);
    //判断两种标签数是否相等  
    if (num_items != num_label)
    {
        mnist_image.close();
        mnist_label.close();
        std::cout << "the image file and label file are not a pair" << std::endl;
    }
    //读图像行数、列数  
    mnist_image.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    mnist_image.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);
    //读取图像  
    std::vector<std::pair<int, cv::Mat>> result(num_items);
    int t = 0;
    for (int i = 0; i < num_items; i++)
    {
        char* pixels = new char[rows * cols];
        mnist_image.read(pixels, rows * cols);
        char label;
        mnist_label.read(&label, 1);
        cv::Mat image(rows, cols, CV_8UC1);
        for (int m = 0; m != rows; m++)
        {
            uchar* ptr = image.ptr<uchar>(m);
            for (int n = 0; n != cols; n++)
            {
                if (pixels[m * cols + n] == 0)
                    ptr[n] = 0;
                else
                    ptr[n] = 255;
            }
        }
        result[t++] = (std::pair<int, cv::Mat>((int)label, std::move(image)));
    }
    mnist_image.close();
    mnist_label.close();
    return result;
}