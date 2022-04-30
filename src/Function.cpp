#include "Function.hpp"

namespace minnet
{
    Tensor MSELoss(const Tensor& pred, const Tensor& real) {
		return (pred - real).pow(2.f).rowsum();
	}

	Tensor NLLLoss(const Tensor& pred, const Tensor& real) {
		return -(real * pred).rowsum();
	}

	Tensor CrossEntropyLoss(const Tensor& pred, const Tensor& real) {
		return NLLLoss(SoftMax(pred).log(), real);
	}

	Tensor Relu(const Tensor& tensor) {
		return tensor.relu();
	}

	Tensor Sigmoid(const Tensor& tensor) {
		return 1.f / ((-tensor).rpow(E) + 1.f);
	}

	Tensor SoftMax(const Tensor& tensor) {
		Tensor exp = (tensor - tensor.max()).rpow(E);
		return exp / exp.rowsum();
	}
} // namespace minnet
