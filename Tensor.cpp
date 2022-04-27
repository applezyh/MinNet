#include "Tensor.hpp"
#include <omp.h>

constexpr float delta = 1e-7;

namespace minnet
{
    void TensorWrong::what() const {
        switch (w_flag)
        {
        case 1:
            std::cout << "Error: out of range( tensor shape: [";
            for (int i = 0; i < shape1.size(); i++) {
                std::cout << shape1[i];
                if (i != shape1.size() - 1) std::cout << ", ";
            }
            std::cout << "], access index: [";
            std::cout << index;
            std::cout << "] )" << std::endl;
        case 2:
            std::cout << "Error: shape conflict( tensor shape: [";
            for (int i = 0; i < shape1.size(); i++) {
                std::cout << shape1[i];
                if (i != shape1.size() - 1) std::cout << ", ";
            }
            std::cout << "], access shape: [";
            for (int i = 0; i < shape2.size(); i++) {
                std::cout << shape2[i];
                if (i != shape2.size() - 1) std::cout << ", ";
            }
            std::cout << "] )" << std::endl;
        default:
            break;
        }
    }

	_Tensor::_Tensor() {}

    _Tensor::_Tensor(const _Tensor& t){
        _shape = t._shape;
        _strided = t._strided;
        _data = t._data;
        _grad = t._grad;
        dy_dx = t.dy_dx;
        _size = t._size;
        operand1 = t.operand1;
        operand2 = t.operand2;
        opeartor = t.opeartor;
        _require_grad = t._require_grad;
    }

    _Tensor::_Tensor(_Tensor&& t) noexcept {
        _shape = std::move(t._shape);
        _strided = std::move(t._strided);
        _data = std::move(t._data);
        _grad = std::move(t._grad);
        dy_dx = t.dy_dx;
        _size = t._size;
        operand1 = t.operand1;
        operand2 = t.operand2;
        opeartor = t.opeartor;
        _require_grad = t._require_grad;
    }

    _Tensor::_Tensor(const std::vector<int> shape) :_size(1) {
        _shape = std::vector<int>(shape);
        for (auto& v : shape) _size *= v;
        _data = std::vector<float>(_size, 0.f);
        _grad = std::vector<float>(_size, 0.f);
        update_strided();
    }

	const std::vector<int> _Tensor::shape() const {
		return _shape;
	}

	int _Tensor::size() const {
		return _size;
	}

	std::vector<float>::iterator _Tensor::begin() {
		return _data.begin();
	}

	std::vector<float>::iterator _Tensor::end() {
		return _data.end();
	}

    std::vector<float>::iterator _Tensor::grad_begin() {
        return _grad.begin();
    }

    std::vector<float>::iterator _Tensor::grad_end() {
        return _grad.end();
    }

    void _Tensor::assignment(float other) {
        for (auto& v : _data) {
            v = other;
        }
    }

    float _Tensor::max() {
        float _max = -INFINITY;
        for (int i = 0; i < _data.size(); i++) {
            if (_max < _data[i]) _max = _data[i];
        }
        return _max;
    }

    float& _Tensor::at(const std::vector<int>& indexs) {
        int index = 0;
        if (indexs.size() != _shape.size()) {
            throw TensorWrong(2, _shape, index);
            return _data[0];
        }
        for (int i = 0; i < indexs.size() - 1; i++) {
            index = (index + indexs[i]) * _shape[i + 1];
        }
        index += indexs[indexs.size() - 1];
        if (index < _size)
            return _data[index];
        throw TensorWrong(1, _shape, index);

    }

    float _Tensor::at(const std::vector<int>& indexs) const {
        int index = 0;
        if (indexs.size() != _shape.size()) {
            throw TensorWrong(2, _shape, index);
            return _data[0];
        }
        for (int i = 0; i < indexs.size() - 1; i++) {
            index = (index + indexs[i]) * _shape[i + 1];
        }
        index += indexs[indexs.size() - 1];
        if (index < _size)
            return _data[index];
        throw TensorWrong(1, _shape, index);

    }

    _Tensor& _Tensor::operator=(const _Tensor& t) {
        _shape = t._shape;
        _strided = t._strided;
        _data = t._data;
        _grad = t._grad;
        dy_dx = t.dy_dx;
        _size = t._size;
        operand1 = t.operand1;
        operand2 = t.operand2;
        opeartor = t.opeartor;
        _require_grad = t._require_grad;
        return *this;
    }

    _Tensor& _Tensor::operator=(_Tensor&& t) noexcept {
        _shape = std::move(t._shape);
        _strided = std::move(t._strided);
        _data = std::move(t._data);
        _grad = std::move(t._grad);
        _size = std::move(t._size);
        dy_dx = t.dy_dx;
        operand1 = t.operand1;
        operand2 = t.operand2;
        opeartor = t.opeartor;
        _require_grad = t._require_grad;
        return *this;
    }

    _Tensor& _Tensor::operator=(float other) {
        for (int i = 0; i < _size; i++) {
            _data[i] = other;
        }
        return *this;
    }

    _Tensor _Tensor::dot(const _Tensor& t) {
        _Tensor* result = &(this->dot2d(t));
        _Tensor ret = std::move(*result);
        delete result;
        ret.require_grad(false);
        return ret;
    }

    _Tensor& operator+(const _Tensor& t, float other) {
        _Tensor& temp = *(new _Tensor(t.shape()));

        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            t.dy_dx->require_grad(false);
            t.dy_dx->assignment(1.f);
            t.opeartor = ADD;
            temp.operand1 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = t._data[i] + other;
        }
        return temp;
    }

    _Tensor& operator+(float other, const _Tensor& t) {
        _Tensor& temp = *(new _Tensor(t.shape()));

        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            t.dy_dx->require_grad(false);
            t.dy_dx->assignment(1.f);
            t.opeartor = ADD;
            temp.operand1 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = t._data[i] + other;
        }
        return temp;
    }

    _Tensor& operator-(const _Tensor& t, float other){
        _Tensor& temp = *(new _Tensor(t.shape()));

        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            t.dy_dx->require_grad(false);
            t.dy_dx->assignment(1.f);
            t.opeartor = SUB;
            temp.operand1 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = t._data[i] - other;
        }
        return temp;
    }

    _Tensor& operator-(float other, const _Tensor& t) {
        _Tensor& temp = *(new _Tensor(t.shape()));

        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            t.dy_dx->require_grad(false);
            t.dy_dx->assignment(-1.f);
            t.opeartor = SUB;
            temp.operand1 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = other - t._data[i];
        }
        return temp;
    }

    _Tensor& operator*(const _Tensor& t, float other){
        _Tensor& temp = *(new _Tensor(t.shape()));

        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            t.dy_dx->require_grad(false);
            t.dy_dx->assignment(other);
            t.opeartor = MUL;
            temp.operand1 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = t._data[i] * other;
        }
        return temp;
    }

    _Tensor& operator*(float other, const _Tensor& t){
        _Tensor& temp = *(new _Tensor(t.shape()));

        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            t.dy_dx->require_grad(false);
            t.dy_dx->assignment(other);
            t.opeartor = MUL;
            temp.operand1 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = t._data[i] * other;
        }
        return temp;
    }

    _Tensor& operator/(const _Tensor& t, float other){
        _Tensor& temp = *(new _Tensor(t.shape()));

        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            t.dy_dx->require_grad(false);
            t.dy_dx->assignment(1 / (other + delta));
            t.opeartor = DIV;
            temp.operand1 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = t._data[i] / (other + delta);
        }
        return temp;
    }

    _Tensor& operator/(float other, const _Tensor& t) {
        _Tensor& temp = *(new _Tensor(t.shape()));

        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            t.dy_dx->require_grad(false);
            for (int i = 0; i < t.dy_dx->size(); i++) {
                t.dy_dx->_data[i] = -1.f * other / (t._data[i] * t._data[i] + delta);
            }
            t.opeartor = DIV;
            temp.operand1 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = other / (t._data[i] + delta);
        }
        return temp;
    }

    _Tensor& _Tensor::operator+(const _Tensor& t) const {
        _Tensor& temp = *(new _Tensor(t.shape()));
        if (!shapeEq(t)) {
            throw TensorWrong(2, shape(), t.shape());
            return temp;
        }  
        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(*this);
            t.dy_dx->assignment(1.f);
            t.dy_dx->require_grad(false);
            t.opeartor = ADD;
            temp.operand2 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(t);
            dy_dx->assignment(1.f);
            dy_dx->require_grad(false);
            opeartor = ADD;
            temp.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        if (!t.require_grad() && !require_grad()) temp.require_grad(false);
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = _data[i] + t._data[i];
        }
        return temp;
        throw TensorWrong(2, shape(), t.shape());
        return temp;
    }

    _Tensor& _Tensor::operator-(const _Tensor& t) const {
        _Tensor& temp = *(new _Tensor(t.shape()));
        if (!shapeEq(t)) {
            throw TensorWrong(2, shape(), t.shape());
            return temp;
        }
        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(*this);
            t.dy_dx->assignment(-1.f);
            t.dy_dx->require_grad(false);
            t.opeartor = SUB;
            temp.operand2 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(t);
            dy_dx->assignment(1.f);
            dy_dx->require_grad(false);
            opeartor = SUB;
            temp.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        if (!t.require_grad() && !require_grad()) temp.require_grad(false);
        
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = _data[i] - t._data[i];
        }
        return temp;
    }

    _Tensor& _Tensor::operator*(const _Tensor& t) const {
        _Tensor& temp = *(new _Tensor(t.shape()));
        if (!shapeEq(t)) {
            throw TensorWrong(2, shape(), t.shape());
            return temp;
        }
        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(*this);
            t.dy_dx->require_grad(false);
            t.opeartor = MUL;
            temp.operand2 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(t);
            dy_dx->require_grad(false);
            opeartor = MUL;
            temp.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        if (!t.require_grad() && !require_grad()) temp.require_grad(false);
        
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = _data[i] * t._data[i];
        }
        return temp;
    }

    _Tensor& _Tensor::operator/(const _Tensor& t) const {
        _Tensor& temp = *(new _Tensor(t.shape()));
        if (!shapeEq(t)) {
            throw TensorWrong(2, shape(), t.shape());
            return temp;
        }
        if (t.require_grad()) {
            _Tensor r1(*this);
            r1.require_grad(false);
            _Tensor r2(t);
            r2.require_grad(false);
            t.dy_dx = std::make_shared<_Tensor>(t.shape());
            for (int i = 0; i < t.size(); i++) {
                t.dy_dx->_data[i] = -1.f * _data[i] / (t._data[i] * t._data[i] + delta);
            }
            t.dy_dx->require_grad(false);
            t.opeartor = DIV;
            temp.operand2 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        if (require_grad()) {
            _Tensor r1 = t;
            r1.require_grad(false);
            _Tensor r2 = t;
            r2.assignment(1.f);
            r2.require_grad(false);
            dy_dx = std::make_shared<_Tensor>(shape());
            for (int i = 0; i < t.size(); i++) {
                dy_dx->_data[i] = 1.f / (t._data[i] + delta);
            }
            dy_dx->require_grad(false);
            opeartor = DIV;
            temp.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        if (!t.require_grad() && !require_grad()) temp.require_grad(false);
        
        for (int i = 0; i < t.size(); i++) {
            temp._data[i] = _data[i] / (t._data[i] + delta);
        }
        return temp;
    }

    _Tensor& _Tensor::operator-() {
        _Tensor& temp = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(shape());
            dy_dx->require_grad(false);
            dy_dx->assignment(-1.f);
            opeartor = SUB;
            temp.operand2 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        else {
            temp.require_grad(false);
        }
        for (int i = 0; i < size(); i++) {
            temp._data[i] = 0.f - _data[i];
        }
        return temp;
    }

    std::ostream& operator<<(std::ostream& out, const _Tensor& tensor) {
        out << "[tensor, shape: [";
        for (int i = 0; i < tensor._shape.size(); i++) {
            out << tensor._shape[i];
            if (i != tensor._shape.size() - 1) out << ", ";
        }
        out << "], type: float32]";
        return out;
    }

    void _Tensor::require_grad(bool grad) const {
        if (_require_grad == grad) return;
        _require_grad = grad;
        if (!_require_grad) {
            _grad = std::vector<float>();
            operand1 = nullptr;
            operand2 = nullptr;
            dy_dx = nullptr;
        }
        else _grad = std::vector<float>(_size, 0.f);
    }

    bool _Tensor::require_grad() const {
        return _require_grad;
    }

    void _Tensor::from_vector_1d(const std::vector<float>& v, int s, int e) {
        if (e == -1) e = v.size();
        if (s < 0 || s > v.size() || e > v.size()) return;
        _data = std::vector<float>(e - s);
        _grad = std::vector<float>(e - s);
        _shape = std::vector<int>{ (int)(e - s)};
        update_strided();
        _size = e - s;
        for (int i = s; i < e; i++) {
            _data[i - s] = v[i];
        }
    }

    void _Tensor::from_vector_2d(const std::vector<std::vector<float>>& v, int s, int e) {
        if (e == -1) e = v.size();
        if (s<0 || s > v.size() || e > v.size()) return;
        if (v.size() == 0) {
            _data.clear();
            _shape.clear();
            _size = 0;
            return;
        }
        _data = std::vector<float>((e - s) * v[0].size());
        _grad = std::vector<float>((e - s) * v[0].size());
        _shape = std::vector<int>{ (int)(e - s), (int)v[0].size() };
        update_strided();
        _size = (e - s) * v[0].size();
        for (int i = s; i < e; i++) {
            for (int j = 0; j < v[0].size(); j++) {
                at(i - s, j) = v[i][j];
            }
        }
    }

    void _Tensor::update_strided() {
        _strided = std::vector<int>(_shape.size());
        _strided[_strided.size() - 1] = 1;
        for (int i = _shape.size() - 2; i >= 0; i--) {
            _strided[i] = _shape[i + 1] * _strided[i + 1];
        }
    }

    bool _Tensor::shapeEq(const _Tensor& t) const {
        if (_shape.size() != t._shape.size()) return false;
        for (int i = 0; i < _shape.size(); i++) {
            if (_shape[i] != t._shape[i]) return false;
        }
        return true;
    }

    bool _Tensor::operator==(const _Tensor& t) const {
        if (!shapeEq(t)) return false;
        for (int i = 0; i < _data.size(); i++) {
            if (_data[i] != t._data[i]) return false;
        }
        return true;
    }

    bool _Tensor::operator!=(const _Tensor& t) const {
        return !(*this == t);
    }

    _Tensor& _Tensor::dot2d(const _Tensor& t) {
        if (_shape.size() != 2 || t._shape.size() != 2
            || _shape[1] != t._shape[0]) 
        {
            throw TensorWrong(2, shape(), t.shape());
            return *this;
        }

        _Tensor trans1 = t, trans2 = *this;
        trans1.transpose(1, 0);
        trans2.transpose(1, 0);

        _Tensor& result = *(new _Tensor(_shape[0], t._shape[1]));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(std::move(trans1));
            dy_dx->require_grad(false);
            opeartor = MATMUL;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        if (t.require_grad()) {
            t.dy_dx = std::make_shared<_Tensor>(std::move(trans2));
            t.dy_dx->require_grad(false);
            t.opeartor = MATMULED;
            result.operand2 = const_cast<_Tensor*>(&t)->shared_from_this();
        }
        #pragma omp parallel for
        for (int i = 0; i < _shape[0]; i++) {
            for (int j = 0; j < t._shape[1]; j++) {
                for (int k = 0; k < _shape[1]; k++) {
                    result.at(i, j) += at(i, k) * t.at(k, j);
                }
            }
        }
        return result;
    }

    _Tensor& _Tensor::pow(float s) const {
        _Tensor& result = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(_shape);
            dy_dx->require_grad(false);
            for (int i = 0; i < dy_dx->size(); i++) {
                dy_dx->_data[i] = s * std::pow(_data[i] + delta, s) / (_data[i] + delta);
            }
            opeartor = POW;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        for (int i = 0; i < size(); i++) {
            result._data[i] = std::pow(_data[i] + delta, s);
        }
        return result;
    }

    _Tensor& _Tensor::rpow(float s) const {
        _Tensor& result = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(_shape);
            dy_dx->require_grad(false);
            for (int i = 0; i < dy_dx->size(); i++) {
                dy_dx->_data[i] = std::log(s + delta) * std::pow(s + delta, _data[i]);
            }
            opeartor = RPOW;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        for (int i = 0; i < size(); i++) {
            result._data[i] = std::pow(s + delta, _data[i]);
        }
        return result;
    }

    _Tensor& _Tensor::log() const {
        _Tensor& result = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(_shape);
            dy_dx->require_grad(false);
            for (int i = 0; i < dy_dx->size(); i++) {
                dy_dx->_data[i] = 1 / (_data[i] + delta);
            }
            opeartor = LOG;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        for (int i = 0; i < size(); i++) {
            result._data[i] = std::log(_data[i] + delta);
        }
        return result;
    }

    _Tensor& _Tensor::mean() const {
        _Tensor& result = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(_shape);
            dy_dx->require_grad(false);
            for (int i = 0; i < dy_dx->size(); i++) {
                dy_dx->_data[i] = 1.f / _data.size();
            }
            opeartor = MEAN;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        float sum = 0.f;
        for (int i = 0; i < size(); i++) {
            sum += _data[i];
        }
        for (int i = 0; i < size(); i++) {
            result._data[i] = sum / _data.size();
        }
        
        return result;
    }

    _Tensor& _Tensor::sum() const {
        _Tensor& result = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(_shape);
            dy_dx->require_grad(false);
            dy_dx->assignment(1.f);
            opeartor = SUM;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        float sum = 0.f;
        for (int i = 0; i < size(); i++) {
            sum += _data[i];
        }
        for (int i = 0; i < size(); i++) {
            result._data[i] = sum;
        }
        return result;
    }

    _Tensor& _Tensor::relu() const {
        _Tensor& result = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(_shape); 
            for (int i = 0; i < dy_dx->size(); i++) {
                dy_dx->_data[i] = _data[i] < 0.f ? 0.f : 1.f;
            }
            dy_dx->require_grad(false);
            opeartor = RELU;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        for (int i = 0; i < size(); i++) {
            result._data[i] = _data[i] < 0.f ? 0.f : _data[i];
        }
        return result;
    }
    _Tensor& _Tensor::rowsum() const {
        _Tensor& result = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(_shape);
            dy_dx->require_grad(false);
            dy_dx->assignment(1.f);
            opeartor = ROWSUM;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        for (int i = 0; i < _shape[0]; i++) {
            float sum = 0.f;
            for (int j = 0; j < _strided[0]; j++) {
                sum += _data[i * _strided[0] + j];
            }
            for (int j = 0; j < _strided[0]; j++) {
                result._data[i * _strided[0] + j] = sum;
            }
        }
        return result;
    }

    _Tensor& _Tensor::colmean() const {
        _Tensor& result = *(new _Tensor(shape()));
        if (require_grad()) {
            dy_dx = std::make_shared<_Tensor>(_shape);
            dy_dx->require_grad(false);
            dy_dx->assignment(1.f);
            opeartor = ROWSUM;
            result.operand1 = const_cast<_Tensor*>(this)->shared_from_this();
        }
        std::vector<float> sum = std::vector<float>(_strided[0]);
        for (int i = 0; i < _shape[0]; i++) {
            for (int j = 0; j < _strided[0]; j++) {
                sum[j] += _data[i * _strided[0] + j] / _shape[0];
            }
        }
        for (int i = 0; i < _shape[0]; i++) {
            for (int j = 0; j < _strided[0]; j++) {
                result._data[j] = sum[j];
            }
        }
        return result;
    }

    void _Tensor::zero_grad() {
        for (auto& v : _grad) {
             v = 0;
        }
        if (operand1) operand1->zero_grad();
        if (operand2) operand2->zero_grad();
    }

    void _Tensor::backward(_Tensor grad) {
        if (!require_grad()) return;
        switch (opeartor)
        {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case SUM:
        case MEAN:
        case LOG:
        case POW:
        case RPOW:
        case RELU:
        case ROWSUM:
            for (int i = 0; i < _grad.size(); i++) {
                grad._data[i] *= dy_dx->_data[i];
            }
            break;
        case MATMUL:
            grad = grad.dot(*dy_dx);
            grad.require_grad(false);

            break;
        case MATMULED:
            grad = dy_dx->dot(grad);
            grad.require_grad(false);
            break;
        default:
            break;
        }
        for (int i = 0; i < _grad.size(); i++) {
                _grad[i] += grad._data[i];
        }
        if (operand1) operand1->backward(grad);
        if (operand2) operand2->backward(grad);
    }

    void _Tensor::backward() {
        _Tensor grad(_shape);
        grad.assignment(1.f);
        grad.require_grad(false);
        backward(grad);
    }
} // namespace minnet
