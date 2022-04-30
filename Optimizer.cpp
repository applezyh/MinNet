#include "Optimizer.hpp"

namespace minnet
{
	SGD::SGD(std::list<Tensor*> parameters, float lr, float b)
	: parameters(parameters), lr(lr), beta(b) {
		momentum = std::list<std::vector<float>>(parameters.size());
		auto params = parameters.begin();
		auto m = momentum.begin();
		for (; params != parameters.end(); params++, m++) {
			*m = std::vector<float>((*params)->size()+(*params)->bias_size(), 0.f);
		}
	}

	void SGD::set_lr(float lr) {
		this->lr = lr;
	}

	float SGD::get_lr() {
		return lr;
	}

	void SGD::step() {
		auto params = parameters.begin();
		auto m = momentum.begin();
		for (; params != parameters.end(); params++, m++) {
			auto param = (*params)->begin();
			auto grad = (*params)->grad_begin();
			int i = 0;
			for (; i < (*params)->size(); i++, param++, grad++) {
				(*m)[i] = beta * (*m)[i] + (1 - beta) * (lr * *grad);
				*param = *param - (*m)[i];
			}
			if ((*params)->bias_size() == 0) continue;
			param = (*params)->bias_begin();
			grad = (*params)->bias_grad_begin();
			for (; i < (*params)->size()+(*params)->bias_size(); i++, param++, grad++) {
				(*m)[i] = beta * (*m)[i] + (1 - beta) * (lr * *grad);
				*param = *param - (*m)[i];
			}
		}
	}
    
} // namespace minnet
