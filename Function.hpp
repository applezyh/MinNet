#pragma once
#include "Tensor.hpp"

namespace minnet
{
    Tensor MSELoss(const Tensor& pred, const Tensor& real);
	Tensor NLLLoss(const Tensor& pred, const Tensor& real);
	Tensor CrossEntropyLoss(const Tensor& pred, const Tensor& real);
	Tensor Relu(const Tensor& tensor);
	Tensor Sigmoid(const Tensor& tensor);
	Tensor SoftMax(const Tensor& tensor);

    template<typename T1, typename T2, typename T3>
    void shuffle(std::vector<T1>* v1, std::vector<T2>* v2 = nullptr, std::vector<T3>* v3 = nullptr) {
        if (!(v1->size() == v2->size() && v2->size() == v3->size() && v1->size() == v3->size())) return;
        for (int i = 0; i < v1->size(); i++) {
            int t1 = rand() % v1->size(), t2 = rand() % v1->size();
            if (v1 != nullptr) {
                auto temp1 = std::move((*v1)[t1]);
                (*v1)[t1] = std::move((*v1)[t2]);
                (*v1)[t2] = std::move(temp1);
            }
            if (v2 != nullptr) {
                auto temp2 = std::move((*v2)[t1]);
                (*v2)[t1] = std::move((*v2)[t2]);
                (*v2)[t2] = std::move(temp2);
            }
            if (v3 != nullptr) {
                auto temp3 = std::move((*v3)[t1]);
                (*v3)[t1] = std::move((*v3)[t2]);
                (*v3)[t2] = std::move(temp3);
            }
        }
    }

    template<typename T>
    concept Iteratable = requires (const T & value) {
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
} // namespace minnet

