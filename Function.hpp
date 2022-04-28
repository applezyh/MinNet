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
} // namespace minnet

