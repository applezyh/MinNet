#include "Optimizer.hpp"
#include <omp.h>

namespace minnet
{
	SGD::SGD(Layer* layer, float lr, float m) {

	}

	void SGD::set_lr(float lr) {
		this->lr = lr;
	}

	float SGD::get_lr() {
		return lr;
	}

	void SGD::step() {

	}
    
} // namespace minnet
