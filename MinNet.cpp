// MinNet.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

//#include "Function.hpp"
#include "mnist.hpp"
#include "Transformation.hpp"
#include "Function.hpp"

#include <iostream>

#define NUM 60000
#define BATCH_SIZE 32

int main() {
    //srand(clock());
    //cv::Mat img = cv::imread("C:\\Users\\apple\\Desktop\\apple\\计算机视觉\\计算机视觉第一次作业\\bird.bmp", 0);
    //img.convertTo(img, CV_32F);
    //cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);
    //cv::imshow("re", img);
    //cv::waitKey(0);
    //cv::destroyWindow("re");
    //minnet::Tensor img1 = minnet::from_mat(img);
    //minnet::Tensor k(3, 11, 11);
    //k.assignment(1.f / (11*11));
    //minnet::Tensor re = img1.padding2d(5);
    //re = re.conv2d(k);
    //img = minnet::to_mat(re);
    //cv::imshow("re", img);
    //cv::waitKey(0);
    //cv::destroyWindow("re");
    //re.backward();
    auto src_data = readAndSave("D:\\BaiduNetdiskDownload\\mnist_dataset\\mnist_dataset\\train-images-idx3-ubyte\\train-images.idx3-ubyte",
        "D:\\BaiduNetdiskDownload\\mnist_dataset\\mnist_dataset\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");

    std::vector<std::vector<float>> data(60000);
    std::vector<std::vector<float>> label(60000);

    for (int i = 0; i < 60000; i++) {
        data[i] = image_to_vec(src_data[i].second);
        label[i] = std::vector<float>(10, 0.f);
        label[i][src_data[i].first] = 1.f;
    }


    minnet::Tensor k1(3 * 784, 10);
    minnet::Tensor b1(1, 10);

    minnet::Tensor kernel(3, 3, 3);

    k1.rand();
    kernel.rand();
    int count = 0;
    float lr = 0.001f;

    for (int i = 0; i < 5 ; i++) {
        minnet::shuffle(&data, &label, &src_data);
        for (int j = 0; j < 60000; j++) {
            minnet::Tensor in;
            in.from_vector_2d(data, j, j + 1);
            minnet::Tensor real;
            real.from_vector_2d(label, j, j + 1);

            minnet::Tensor result = in.reshape(28, 28, 1);
            result = result.padding2d(1);
            result = result.conv2d(kernel);
            result = result.reshape(1, 3 * 784);
            result = minnet::Relu(result);
            result = result.dot2d(k1) + b1;
            real.reshape(1, 10);
            minnet::Tensor loss = minnet::CrossEntropyLoss(result, real);

            loss.zero_grad();
            loss.backward();

            for (auto it1 = k1.begin(), it2 = k1.grad_begin(); it1 != k1.end(); it1++, it2++) {
                *it1 = *it1 - lr * *it2;
            }
            for (auto it1 = b1.begin(), it2 = b1.grad_begin(); it1 != b1.end(); it1++, it2++) {
                *it1 = *it1 - lr * *it2;
            }
            for (auto it1 = kernel.begin(), it2 = kernel.grad_begin(); it1 != kernel.end(); it1++, it2++) {
                *it1 = *it1 - lr * *it2;
            }
        }
        if ((i + 1) % 5 == 0) {
            lr /= 2;
        }
        std::cout << "epoch: " << i + 1 << std::endl;
    }
    count = 0;
    for (int j = 0; j < 60000; j++) {
        minnet::Tensor in;
        in.from_vector_2d(data, j, j + 1);
        minnet::Tensor real;
        real.from_vector_2d(label, j, j + 1);
        minnet::Tensor result = in.reshape(28, 28, 1);
        result = result.padding2d(1);
        result = result.conv2d(kernel);
        result = result.reshape(1, 3 * 784);
        result = minnet::Relu(result);
        result = result.dot2d(k1) + b1;
        real.reshape(1, 10);
        if (j < 20) {
            cv::Mat temp;
            cv::resize(src_data[j].second, temp, cv::Size(224, 224));
            cv::imshow("result", temp);
            std::cout << minnet::argMax(result) << std::endl;
            cv::waitKey(0); 
        }
        if (minnet::argMax(result) == minnet::argMax(real)) count++;
    }
    cv::destroyWindow("result");
    std::cout << "after train: " << count / 60000.f << std::endl;
    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
