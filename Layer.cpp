#include "Layer.hpp"

namespace minnet
{
	Layer::Layer(int input, int output) {
	}

	LinerLayer::LinerLayer(int input, int output) :Layer(input, output) {}

	Tensor LinerLayer::Forward(Tensor& tensor) {

	}
} // namespace minnet
