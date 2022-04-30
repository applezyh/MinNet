// MinNet.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "Dataset.hpp"
#include "Transformation.hpp"
#include "Function.hpp"
#include "Layer.hpp"
#include "Optimizer.hpp"

#include <iostream>

class Net :public minnet::Model {
public:
    Net() {
        conv1 = minnet::Conv2d(3, 1);
        conv2 = minnet::Conv2d(3, 3);
        fc1 = minnet::Linear(3 * (7 * 7), 10);
        add_layer(&conv1);
        add_layer(&conv2);
        add_layer(&fc1);
    }
    minnet::Tensor Forward(minnet::Tensor input) override {
        minnet::Tensor out = conv1(input);
        out = minnet::Relu(out);
        out = out.maxpool2d();
        out = conv2(out);
        out = minnet::Relu(out);
        out = out.maxpool2d();
        out = out.reshape(1, -1);
        return fc1(out);
    }
private:
    minnet::Conv2d conv1;
    minnet::Conv2d conv2;
    minnet::Linear fc1;
};

#define NUM 60000
#define BATCH_SIZE 32

#define SHOW_RESULT

int main() {
    srand(clock());
    auto src_data = load_mnist("D:\\BaiduNetdiskDownload\\mnist_dataset\\mnist_dataset\\train-images-idx3-ubyte\\train-images.idx3-ubyte", 
                                "D:\\BaiduNetdiskDownload\\mnist_dataset\\mnist_dataset\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");
   // auto src_data = load_cifa10("C:\\Users\\apple\\Downloads\\cifar-10-binary\\cifar-10-batches-bin");
    std::vector<std::vector<float>> data(NUM);
    std::vector<std::vector<float>> label(NUM);

    for (int i = 0; i < NUM; i++) {
        data[i] = image_to_vec(src_data[i].second);
        label[i] = std::vector<float>(10, 0.f);
        label[i][src_data[i].first] = 1.f;
    }

    //std::string classes[] = { "airplane" , "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
    
    Net net;
    float lr = 0.0004f;
    minnet::SGD opt(net.parameters(), lr);
    
    for (int i = 0; i < 5 ; i++) {
        minnet::shuffle(&data, &label, &src_data);
        for (int j = 0; j < NUM; j++) {
            minnet::Tensor in;
            in.from_vector_2d(data, j, j + 1);
            minnet::Tensor real;
            real.from_vector_2d(label, j, j + 1);
            minnet::Tensor result = in.reshape(28, 28, 1);
            result = net(result);
            real.reshape(1, 10);
            minnet::Tensor loss = minnet::CrossEntropyLoss(result, real);
            loss.zero_grad();
            loss.backward();
            opt.step();
        }
        std::cout << "epoch: " << i + 1 << std::endl;
    }
#ifdef SHOW_RESULT
    int count = 0;
    for (int j = 0; j < NUM; j++) {
        minnet::Tensor in;
        in.from_vector_2d(data, j, j + 1);
        minnet::Tensor real;
        real.from_vector_2d(label, j, j + 1);
        minnet::Tensor result = in.reshape(28, 28, 1);
        result = net(result);
        real.reshape(1, 10);
        if (j < 20) {
            cv::Mat temp;
            cv::resize(src_data[j].second, temp, cv::Size(224, 224));
            cv::imshow("result", temp);
            std::cout << "pred: " << minnet::argMax(result) << " real: ";
            std::cout << minnet::argMax(real) << std::endl;
            cv::waitKey(0); 
            cv::destroyWindow("result");
        }
        if (minnet::argMax(result) == minnet::argMax(real)) count++;
    }
    std::cout << "after train: " << count / (NUM * 1.f) << std::endl;
#endif // SHOW_RESULT
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
