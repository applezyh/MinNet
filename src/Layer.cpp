#include "Layer.hpp"

namespace minnet
{
	Tensor Layer::operator()(Tensor& input)
	{
		return Forward(input);
	}
	void Layer::train() {
		_train = 1;
	}
	void Layer::eval() {
		_train = 0;
	}
	Linear::Linear(int input, int output) {
		param = Tensor(input, output);
		bias = Tensor(1, output);
		param.rand();
	}
	Tensor Linear::Forward(Tensor& tensor) {
		return tensor.dot2d(param) + bias;
	}
	std::vector<Tensor*> Linear::get_param() {
		return std::vector<Tensor*>{&param, &bias};
	}

	Conv2d::Conv2d(int out_ch, int in_ch, int kernel_size, int padding, int stride_x, int stride_y)
	: padding(padding), stride_x(stride_x), stride_y(stride_y) {
		param = Tensor(out_ch, in_ch, kernel_size, kernel_size);
		param.rand();
		param.set_conv_bias(out_ch);
	}
	Tensor Conv2d::Forward(Tensor& tensor) {
		return tensor.conv2d(param, padding, stride_x, stride_y);
	}
	std::vector<Tensor*> Conv2d::get_param() {
		return std::vector<Tensor*>{&param};
	}

	DropOut::DropOut(float proportion) : proportion(proportion) {

	}

	Tensor DropOut::Forward(Tensor& tensor) {
		if (_train) return tensor.dropout(proportion);
		else return tensor;
	}

	std::vector<Tensor*> DropOut::get_param() {
		return std::vector<Tensor*>();
	}

	void Model::add_layer(Layer* layer) {
		layer_list.push_back(layer);
	}
	std::list<Tensor*>  Model::parameters() {
		std::list<Tensor*> ret;
		for (auto& layer : layer_list) {
			for (auto& tensor : layer->get_param()) {
				ret.push_back(tensor);
			}
		}
		return ret;
	}
	Tensor  Model::operator()(Tensor input) {
		return Forward(input);
	}
	void Model::train() {
		for (auto& layer : layer_list) layer->train();
	}
	void Model::eval() {
		for (auto& layer : layer_list) layer->eval();
	}

} // namespace minnet
