#ifndef OPT_H_
#define OPT_H_
#include <list>
#include <vector>

#include "Tensor.hpp"

namespace minnet
{

	class SGD  {
	public:
		SGD(std::list<Tensor*> parameters, float lr = 0.001, float b = 0.f);

		void set_lr(float lr);
		float get_lr();

		void step();
	private:
		std::list<std::vector<float>> momentum = {};
		std::list<Tensor*> parameters = {};
		float lr = 0.f;
		float beta = 0.f;
	};
} // namespace minnet
#endif // OPT_H_

