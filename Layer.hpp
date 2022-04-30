#ifndef LAYER_H_
#define LAYER_H_
#include "Tensor.hpp"
#include <list>

namespace minnet
{
	class Layer {
	public:
		virtual Tensor Forward(Tensor& tensor) = 0;
		virtual std::vector<Tensor*> get_param() = 0;
		Tensor operator()(Tensor& input);
		Tensor param;
	};

	class Linear : public Layer {
	public:
		Linear() {}
		Linear(int input, int output);
		Tensor Forward(Tensor& tensor) override;
		std::vector<Tensor*> get_param() override;
	private:
		Tensor bias;
	};

	class Conv2d : public Layer {
	public:
		Conv2d() {}
		Conv2d(int out_ch, int in_ch, int kernel_size = 3, int padding = 1, int stride_x = 1, int stride_y = 1);
		Tensor Forward(Tensor& tensor) override;
		std::vector<Tensor*> get_param() override;
	private:
		int padding, stride_x, stride_y;
	};

	class Model {
	public:
		void push_layer(Layer* layer);
		std::list<Tensor*> parameters();
		Tensor operator()(Tensor& input);
		
		virtual Tensor Forward(Tensor& input) = 0;
	private:
		std::list<Layer*> layer_list;
	};
} // namespace minnet
#endif // LAYER_H_


