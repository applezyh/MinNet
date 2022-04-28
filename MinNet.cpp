// MinNet.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "Function.hpp"
#include "mnist.hpp"

#include <iostream>

#define NUM 60000
#define BATCH_SIZE 32

template<typename T1, typename T2, typename T3>
void shuffle(std::vector<T1>& v1, std::vector<T2>& v2, std::vector<T3>& v3) {
    if (!(v1.size() == v2.size() && v2.size() == v3.size() && v1.size() == v3.size())) return;
    for (int i = 0; i < v1.size(); i++) {
        int t1 = rand() % v1.size(), t2 = rand() % v1.size();

        auto temp1 = std::move(v1[t1]);
        v1[t1] = std::move(v1[t2]);
        v1[t2] = std::move(temp1);

        auto temp2 = std::move(v2[t1]);
        v2[t1] = std::move(v2[t2]);
        v2[t2] = std::move(temp2);

        auto temp3 = std::move(v3[t1]);
        v3[t1] = std::move(v3[t2]);
        v3[t2] = std::move(temp3);
    }
}

template<typename T> 
concept Iteratable = requires (const T& value) {
    value.begin();
    value.end();
};

template<typename T> requires Iteratable<T>
int argMax(const T& v) {
    if (v.begin() == v.end()) return -1;
    float max = *(v.begin());
    int index = 0;
    int i = 0;
    for (auto it = v.begin(); it != v.end(); it++, i++) {
        if (max < *it) max = *it, index = i;
    }
    return index;
}


int main() {
    srand(clock());
    auto src_data = readAndSave("D:\\BaiduNetdiskDownload\\mnist_dataset\\mnist_dataset\\train-images-idx3-ubyte\\train-images.idx3-ubyte",
        "D:\\BaiduNetdiskDownload\\mnist_dataset\\mnist_dataset\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");

    std::vector<std::vector<float>> data(60000);
    std::vector<std::vector<float>> label(60000);

    for (int i = 0; i < 60000; i++) {
        data[i] = image_to_vec(src_data[i].second);
        label[i] = std::vector<float>(10, 0.f);
        label[i][src_data[i].first] = 1.f;
    }


    minnet::Tensor k1(784, 10);
    minnet::Tensor b1(1, 10);

    minnet::Tensor k2(10, 10);
    minnet::Tensor b2(1, 10);
    k1.rand();
    k2.rand();
    int count = 0;
    float lr = 0.001f;
    ::shuffle(data, label, src_data);
    for (int i = 0; i < 5 ; i++) {
        ::shuffle(data, label, src_data);
        for (int j = 0; j < 60000; j++) {
            minnet::Tensor in;
            in.from_vector_2d(data, j, j + 1);
            minnet::Tensor real;
            real.from_vector_2d(label, j, j + 1);

            minnet::Tensor result = in.dot2d(k1) + b1;
            result = minnet::Relu(result);
            result = result.dot2d(k2) + b2;
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
            for (auto it1 = k2.begin(), it2 = k2.grad_begin(); it1 != k2.end(); it1++, it2++) {
                *it1 = *it1 - lr * *it2;
            }
            for (auto it1 = b2.begin(), it2 = b2.grad_begin(); it1 != b2.end(); it1++, it2++) {
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
        minnet::Tensor result = in.dot2d(k1) + b1;
        result = minnet::Relu(result);
        result = result.dot2d(k2) + b2;
        if (j < 20) {
            cv::Mat temp;
            cv::resize(src_data[j].second, temp, cv::Size(224, 224));
            cv::imshow("result", temp);
            std::cout << argMax(result) << std::endl;
            cv::waitKey(0);
            cv::destroyWindow("result");
        }
        if (argMax(result) == argMax(real)) count++;
    }
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
