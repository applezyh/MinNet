// MinNet.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "Dataset.hpp"
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
        out = minnet::MaxPool2d(out);
        out = conv2(out);
        out = minnet::Relu(out);
        out = minnet::MaxPool2d(out);
        out = out.reshape(1, -1);
        return fc1(out);
    }
private:
    minnet::Conv2d conv1;
    minnet::Conv2d conv2;
    minnet::Linear fc1;
};

#define NUM 60000

#define SHOW_RESULT

int main(int argc, char**argv) {
    if(argc != 3){
        std::cout<<"must input mnist dataset data path and label path"<<std::endl;
        return 0;
    }
    srand(clock());
    auto src_data = load_mnist(argv[1], argv[2]);

    if (src_data.size() != 60000) {
        std::cout<<"read dataset erro!"<<std::endl;
        return 0;
    }

    std::vector<std::vector<float>> data(NUM);
    std::vector<std::vector<float>> label(NUM);

    for (int i = 0; i < NUM; i++) {
        data[i] = image_to_vec(src_data[i].second);
        label[i] = std::vector<float>(10, 0.f);
        label[i][src_data[i].first] = 1.f;
    }
    
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
