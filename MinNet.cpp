// MinNet.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "Function.hpp"
#include "mnist.hpp"

#include <iostream>

#define NUM 60000
#define BATCH_SIZE 32

template<typename T1, typename T2, typename T3>
void shuffle(std::vector<T1>& v1, std::vector<T2>& v2, std::vector<T3>& v3) {
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
int argMax(std::vector<T>& v) {
    T& max = v[0];
    int index = 0;
    for (int i = 0; i < v.size(); i++) {
        if (max < v[i]) max = v[i], index = i;
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


    minnet::Tensor k(784, 10);
    minnet::Tensor b(1, 10);
    k.rand();
    int count = 0;
    ::shuffle(data, label, src_data);
    for (int j = 0; j < 1000; j++) {
        minnet::Tensor in;
        in.from_vector_2d(data, j, j + 1);
        minnet::Tensor real;
        real.from_vector_2d(label, j, j + 1);
        minnet::Tensor result = in.dot2d(k) + b;
        real.reshape(1, 10);
        float max = -1.f;
        int index = 0;
        int i = 0;
        for (auto it1 = result.begin(); it1 != result.end(); it1++) {
            if (*it1 > max) max = *it1, index = i;
            i++;
        }
        max = -1.f;
        int real_index = 0;
        i = 0;
        for (auto it1 = real.begin(); it1 != real.end(); it1++) {
            if (*it1 > max) max = *it1, real_index = i;
            i++;
        }
        if (index == real_index) count++;
    }
    std::cout <<"without train: " << count / 1000.f << std::endl;

    for (int j = 0; j < 10; j++) {
        minnet::Tensor in;
        in.from_vector_2d(data, j, j + 1);
        minnet::Tensor real;
        real.from_vector_2d(label, j, j + 1);
        minnet::Tensor result = in.dot2d(k) + b;
        real.reshape(1, 10);
        float max = -1.f;
        int index = 0;
        int i = 0;
        for (auto it1 = result.begin(); it1 != result.end(); it1++) {
            if (*it1 > max) max = *it1, index = i;
            i++;
        }
        cv::Mat temp;
        cv::resize(src_data[j].second, temp, cv::Size(224, 224));
        cv::imshow("result", temp);
        std::cout << index << std::endl;
        cv::waitKey(0);
    }
    cv::destroyWindow("result");

    for (int i = 0; i < 5 ; i++) {
        ::shuffle(data, label, src_data);
        for (int j = 0; j < 60000; j++) {
            minnet::Tensor in;
            in.from_vector_2d(data, j, j + 1);
            minnet::Tensor real;
            real.from_vector_2d(label, j, j + 1);

            minnet::Tensor result = in.dot2d(k) + b;
            real.reshape(1, 10);
            minnet::Tensor loss = minnet::CrossEntropyLoss(result, real);

            loss.zero_grad();
            loss.backward();

            for (auto it1 = k.begin(), it2 = k.grad_begin(); it1 != k.end(); it1++, it2++) {
                *it1 = *it1 - 0.001f * *it2;
            }
            for (auto it1 = b.begin(), it2 = b.grad_begin(); it1 != b.end(); it1++, it2++) {
                *it1 = *it1 - 0.001f * *it2;
            }
        }
        std::cout << "epoch: " << i + 1 << std::endl;
    }
    for (int j = 0; j < 20; j++) {
        minnet::Tensor in;
        in.from_vector_2d(data, j, j + 1);
        minnet::Tensor real;
        real.from_vector_2d(label, j, j + 1);
        minnet::Tensor result = in.dot2d(k) + b;
        real.reshape(1, 10);
        float max = -1.f;
        int index = 0;
        int i = 0;
        for (auto it1 = result.begin(); it1 != result.end(); it1++) {
            if (*it1 > max) max = *it1, index = i;
            i++;
        }
        cv::Mat temp;
        cv::resize(src_data[j].second, temp, cv::Size(224, 224));
        cv::imshow("result", temp);
        std::cout << index << std::endl;
        cv::waitKey(0);
    }
    cv::destroyWindow("result");
    count = 0;
    ::shuffle(data, label, src_data);
    for (int j = 0; j < 60000; j++) {
        minnet::Tensor in;
        in.from_vector_2d(data, j, j + 1);
        minnet::Tensor real;
        real.from_vector_2d(label, j, j + 1);
        minnet::Tensor result = in.dot2d(k) + b;
        real.reshape(1, 10);
        float max = -1.f;
        int index = 0;
        int i = 0;
        for (auto it1 = result.begin(); it1 != result.end(); it1++) {
            if (*it1 > max) max = *it1, index = i;
            i++;
        }
        max = -1.f;
        int real_index = 0;
        i = 0;
        for (auto it1 = real.begin(); it1 != real.end(); it1++) {
            if (*it1 > max) max = *it1, real_index = i;
            i++;
        }
        if (index == real_index) count++;
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
