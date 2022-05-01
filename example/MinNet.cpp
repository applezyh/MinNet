// MinNet.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "Util.hpp"
#include "Dataset.hpp"
#include "Function.hpp"
#include "Layer.hpp"
#include "Optimizer.hpp"

#include <iostream>

class Net :public minnet::Model {
public:
    Net() {
        conv1 = minnet::Conv2d(1, 1);
        conv2 = minnet::Conv2d(1, 1);
        fc1 = minnet::Linear(1 * (7 * 7), 10);
        dropout = minnet::DropOut(0.5);
        add_layer(&conv1);
        add_layer(&conv2);
        add_layer(&fc1);
        add_layer(&dropout);
    }
    minnet::Tensor Forward(minnet::Tensor input) override {
        minnet::Tensor out = conv1(input);
        out = dropout(out);
        out = minnet::Relu(out);
        out = minnet::MaxPool2d(out);
        out = conv2(out);
        out = dropout(out);
        out = minnet::Relu(out);
        out = minnet::MaxPool2d(out);
        out = out.reshape(1, -1);
        return fc1(out);
    }
private:
    minnet::Conv2d conv1;
    minnet::Conv2d conv2;
    minnet::Linear fc1;
    minnet::DropOut dropout;
};

#define NUM 60000
#define TEST_NUM 10000

#define SHOW_RESULT

int main(int argc, char**argv) {
    if(argc != 5){
        std::cout<<"must input mnist dataset data path and label path"<<std::endl;
        return 0;
    }

    auto src_data = load_mnist(argv[1], argv[2]);
    auto test_data = load_mnist(argv[3], argv[4]);

    if (src_data.size() != 60000 || test_data.size() != 10000) {
        std::cout<<"read dataset erro!"<<std::endl;
        return 0;
    }
    srand(clock());
    std::vector<std::vector<float>> data(NUM);
    std::vector<std::vector<float>> label(NUM);

    std::vector<std::vector<float>> test(TEST_NUM);
    std::vector<std::vector<float>> test_label(TEST_NUM);

    for (int i = 0; i < NUM; i++) {
        data[i] = image_to_vec(src_data[i].second);
        label[i] = std::vector<float>(10, 0.f);
        label[i][src_data[i].first] = 1.f;
    }

    for (int i = 0; i < TEST_NUM; i++) {
        test[i] = image_to_vec(test_data[i].second);
        test_label[i] = std::vector<float>(10, 0.f);
        test_label[i][test_data[i].first] = 1.f;
    }
    
    Net net;
    float lr = 0.0004f;
    minnet::SGD opt(net.parameters(), lr, 0.9f);
    Timer timer;
    net.train();
    for (int i = 0; i < 10 ; i++) {
        minnet::shuffle(&data, &label, &src_data);
        float forward_time = 0.f;
        float backward_time = 0.f;
        for (int j = 0; j < NUM; j++) {
            minnet::Tensor in;
            in.from_vector_2d(data, j, j + 1);
            minnet::Tensor real;
            real.from_vector_2d(label, j, j + 1);
            minnet::Tensor result = in.reshape(28, 28, 1);
            timer.begin();
            result = net(result);
            real.reshape(1, 10);
            minnet::Tensor loss = minnet::CrossEntropyLoss(result, real);
            timer.end();
            forward_time += timer.cost();
            timer.begin();
            loss.zero_grad();
            loss.backward();
            opt.step();
            timer.end();
            backward_time += timer.cost();
        }
        std::cout << "epoch: " << i + 1 << " forward cost: " << forward_time << " backward cost: " << backward_time << std::endl;
    }
#ifdef SHOW_RESULT
    net.eval();
    int count = 0;
    for (int j = 0; j < TEST_NUM; j++) {
        minnet::Tensor in;
        in.from_vector_2d(test, j, j + 1);
        minnet::Tensor real;
        real.from_vector_2d(test_label, j, j + 1);
        minnet::Tensor result = in.reshape(28, 28, 1);
        result = net(result);
        real.reshape(1, 10);
        if (j < 20) {
            cv::Mat temp;
            cv::resize(test_data[j].second, temp, cv::Size(224, 224));
            cv::imshow("result", temp);
            std::cout << "pred: " << minnet::argMax(result) << " real: ";
            std::cout << minnet::argMax(real) << std::endl;
            cv::waitKey(0); 
            cv::destroyWindow("result");
        }
        if (minnet::argMax(result) == minnet::argMax(real)) count++;
    }
    std::cout << "after train with eval: " << count / (TEST_NUM * 1.f) << std::endl;
#endif // SHOW_RESULT
    return 0;
}
