#include "Layer.hpp"

namespace minnet
{
	Layer::Layer(int input, int output) {
	}

	LinerLayer::LinerLayer(int input, int output) :Layer(input, output) {}

	Tensor LinerLayer::Forward(Tensor& tensor) {

	}

	Tensor MSELoss(Tensor& pred, Tensor& real) {

	}

	Tensor CrossEntropyLoss(Tensor& pred, Tensor& real) {

	}

	Tensor Relu(Tensor& tensor) {

	}

	Tensor Sigmoid(Tensor& tensor) {

	}

	Tensor SoftMax(Tensor& tensor) {

	}
} // namespace minnet
