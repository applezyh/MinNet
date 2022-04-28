#pragma once
#include "Tensor.hpp"

#include <random>

namespace minnet
{
	
	Tensor MSELoss(Tensor& pred, Tensor& real);
	Tensor CrossEntropyLoss(Tensor& pred, Tensor& real);
	Tensor Relu(Tensor& tensor);
	Tensor Sigmoid(Tensor& tensor);
	Tensor SoftMax(Tensor& tensor);
	class Layer {
	public:
		Layer() = delete;
		Layer(int input, int output);
		virtual std::vector<Tensor> Forward(std::vector<Tensor>& tensor) = 0;
		Tensor param;
		Tensor bias;
	};

	class LinerLayer : public Layer {
	public:
		LinerLayer(int input, int output);
		std::vector<Tensor> Forward(std::vector<Tensor>& tensor) override;
	private:
	};
} // namespace minnet


