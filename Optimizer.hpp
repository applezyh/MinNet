#pragma once
#include "Layer.hpp"

namespace minnet
{

	class SGD  {
	public:
		SGD(Layer* layer, float lr = 0.001, float m = 0.f);

		void set_lr(float lr);
		float get_lr();

		void step();
	private:
		std::vector<float> momentum = {};
		Layer* layer_ptr = nullptr;
		float lr = 0.f;
		float beta = 0.f;
	};
} // namespace minnet


