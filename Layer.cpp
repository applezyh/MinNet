#include "Layer.hpp"

namespace minnet
{
	Layer::Layer(int input, int output) {
		param = Tensor(input, output);
		bias = Tensor(output, output);
	}

	LinerLayer::LinerLayer(int input, int output) :Layer(input, output) {}

	std::vector<Tensor> LinerLayer::Forward(std::vector<Tensor>& tensors) {
		std::vector<Tensor> result;
		for (auto& tensor : tensors) {
			result.push_back(tensor.dot2d(param) + bias);
		}
		return result;
	}
} // namespace minnet
