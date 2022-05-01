#ifndef TENSOR_H_
#define TENSOR_H_

#include <memory>
#include <iostream>
#include <vector>



namespace minnet
{
    constexpr float E  = 2.718281828459045235360287471352662497757247093f;

    enum {
        ADD,
        SUB,
        MUL,
        DIV,
        POW,
        LOG,
        RPOW,
        SUM,
        MEAN,
        MATMUL,
        ROWSUM,
        RELU,
        CONV2D,
        PADDING,
        RESHAPE,
        MAXPOOL,
        DROPOUT
    };


    class TensorWrong {
    public:
        TensorWrong(int flag) :w_flag(flag), index(-1) {}
        TensorWrong(int flag, const std::vector<int>& shape1, const std::vector<int>& shape2) :w_flag(flag), shape1(shape1), shape2(shape2), index(-1) {}
        TensorWrong(int flag, const std::vector<int>& shape1, int index) :w_flag(flag), shape1(shape1), index(index) {}
    public:
        void what() const;
    private:
        std::vector<int> shape1;
        std::vector<int> shape2;
        int index;
        int w_flag;
    };

    class _Tensor : public std::enable_shared_from_this<_Tensor> {
    public:
        _Tensor();
        _Tensor(const _Tensor& t);
        _Tensor(_Tensor&& t) noexcept;
        _Tensor(const std::vector<int> shape);

        template<typename T, typename... Args> requires (std::is_same_v<T, int> || std::is_same_v<T, size_t>)
        _Tensor(const T& arg, const Args&... args) :_size(1) {
            Init(arg, args...);
            _data = std::vector<float>(_size, 0.f);
            _grad = std::vector<float>(_size, 0.f);
            update_strided();
        }

        template<typename T, typename... Args>  
        void Init(const T& arg, const Args&... args) {
            _size *= arg;
            _shape.push_back(arg);
            Init(args...);
        }

        void Init(const int& arg) {
            _size *= arg;
            _shape.push_back(arg);
            return;
        }

        template<typename T, typename... Args>  requires (std::is_same_v<T, int> || std::is_same_v<T, size_t>)
        _Tensor& reshape(const T& arg, const Args&... args) {         
            std::vector<int> new_shape;
            getIndex(new_shape, arg, args...);
            int new_size = 1;
            for (auto& v : new_shape) new_size *= v;
            if (new_size == _size) {
                _Tensor& temp = *(new _Tensor(*this));
                temp._shape = new_shape;
                temp.update_strided();
                temp.operand1 = this->shared_from_this();
                temp.opeartor = RESHAPE;
                return temp;
            }
            else {
                int count = 0;
                int temp_size = 1;
                for (int i = 0; i < new_shape.size(); i++) {
                    if (new_shape[i] == -1) count++;
                    else temp_size *= new_shape[i];
                }
                if (temp_size != 0 && count == 1 && _size % temp_size == 0) {
                    int t = _size / temp_size;
                    for (int i = 0; i < new_shape.size(); i++) {
                        if (new_shape[i] == -1) new_shape[i] = t;
                    }
                    _Tensor& temp = *(new _Tensor(*this));
                    temp._shape = new_shape;
                    temp.update_strided();
                    temp.operand1 = this->shared_from_this();
                    temp.opeartor = RESHAPE;
                    return temp;
                }
                throw TensorWrong(2, _shape, new_shape);
                return *this;
            }
        }

        template<typename... Args>
        float& at(const Args&... args) {
            int index = 0;
            int i = 0;
            _Tensor::getIndex(index, i, args...);
            if (index < _size)
                return _data[index];
            throw TensorWrong(1, _shape, index);
            return _data[index];
        }   

        template<typename... Args>
        float at(const Args&... args) const {
            int index = 0;
            int i = 0;
            _Tensor::getIndex(index, i, args...);
            if(index<_size)
                return _data[index];
            throw TensorWrong(1, _shape, index);

        }

        template<typename T, typename... Args> requires (std::is_same_v<T, int> || std::is_same_v<T, size_t>)
            void transpose(const T& arg, const Args&... args) {
            std::vector<int> trans;
            getIndex(trans, arg, args...);
            if (trans.size() != _shape.size()) {
                throw TensorWrong(2, trans, _shape);
                return;
            }
            std::vector<int> map(_shape.size(), 0);
            for (int i = 0; i < trans.size(); i++) {
                if (trans[i] > _shape.size()) {
                    throw TensorWrong(2, trans, _shape);
                    return;
                }
                map[trans[i]]++;
                if (map[trans[i]] > 1) {
                    throw TensorWrong(2, trans, _shape);
                    return;
                }
            }
            std::vector<int> new_shape;
            for (int i = 0; i < trans.size(); i++) {
                new_shape.push_back(_shape[trans[i]]);
            }
            std::vector<int> new_strided;
            for (int i = 0; i < trans.size(); i++) {
                new_strided.push_back(_strided[trans[i]]);
            }
            _shape = std::move(new_shape);
            _strided = std::move(new_strided);
        }

        float& at(const std::vector<int>& indexs);
        float at(const std::vector<int>& indexs) const;

        const std::vector<int> shape() const;
        int size() const;

        std::vector<float>::iterator begin();
        std::vector<float>::iterator end();

        std::vector<float>::iterator grad_begin();
        std::vector<float>::iterator grad_end();

        std::vector<float>::iterator bias_begin();
        std::vector<float>::iterator bias_end();

        std::vector<float>::iterator bias_grad_begin();
        std::vector<float>::iterator bias_grad_end();

        int bias_size();
        void set_conv_bias(int num);

        void assignment(float other);
        void rand();

        float max();

        

        _Tensor& operator=(const _Tensor& t);
        _Tensor& operator=(_Tensor&& t) noexcept;
        _Tensor& operator=(float other);

        friend _Tensor& operator+(const _Tensor& t, float other);
        friend _Tensor& operator+(float other, const _Tensor& t);

        friend _Tensor& operator-(const _Tensor& t, float other);
        friend _Tensor& operator-(float other, const _Tensor& t);

        friend _Tensor& operator*(const _Tensor& t, float other);
        friend _Tensor& operator*(float other, const _Tensor& t);

        friend _Tensor& operator/(const _Tensor& t, float other);
        friend _Tensor& operator/(float other, const _Tensor& t);

        _Tensor& operator+(const _Tensor& t) const;
        _Tensor& operator-(const _Tensor& t) const;
        _Tensor& operator*(const _Tensor& t) const;
        _Tensor& operator/(const _Tensor& t) const;

        _Tensor& operator-();

        bool operator==(const _Tensor& t) const;
        bool operator!=(const _Tensor& t) const;

        friend std::ostream& operator<<(std::ostream& out, const _Tensor& tensor);

        void from_vector_1d(const std::vector<float>& v, int s = 0, int e = -1);
        void from_vector_2d(const std::vector<std::vector<float>>& v, int s = 0, int e = -1);

        _Tensor& dropout(float proportion) const;
        _Tensor& maxpool2d(size_t size) const;
        _Tensor& padding2d(size_t size) const;
        _Tensor& conv2d(const _Tensor& kernel, int stride_x, int stride_y) const;
        _Tensor& mean() const;
        _Tensor& dot2d(const _Tensor& other);
        _Tensor& pow(float s) const;
        _Tensor& rpow(float s) const;
        _Tensor& log(float s) const;
        _Tensor& sum() const;
        _Tensor& relu() const;
        _Tensor& rowsum() const;
        _Tensor& colmean() const;

        void require_grad(bool grad) const;
        bool require_grad() const;

        void zero_grad();
        void backward(_Tensor grad, const std::vector<float>& dy_dx);
        void backward();

    private:
        _Tensor dot(const _Tensor& t);

        template<typename T, typename... Args> requires (std::is_same_v<T, int> || std::is_same_v<T, size_t>)
        void getIndex(int& index, int& i, const T& arg, const Args&... args) const {
            index += arg * _strided[i++];
            getIndex(index, i, args...);
        }

        void getIndex(int& index, int& i) const {
            return;
        }

        template<typename T, typename... Args> requires (std::is_same_v<T, int> || std::is_same_v<T, size_t>)
            void getIndex(std::vector<int>& index, const T& arg, const Args&... args) const {
            index.push_back(arg);
            getIndex(index, args...);
        }

        template<typename T> requires (std::is_same_v<T, int> || std::is_same_v<T, size_t>)
            void getIndex(std::vector<int>& index, const T& arg) const {
            index.push_back(arg);
            return;
        }

        void transpose(int i, _Tensor& new_tensor, const std::vector<int>& trans, std::vector<int>& index) {
            if (i == _shape.size()) {
                std::vector<int> new_index;
                for (int i = 0; i < trans.size(); i++) {
                    new_index.push_back(index[trans[i]]);
                }
                new_tensor.at(new_index) = at(index);
                return;
            }
            for (int j = 0; j < _shape[i]; j++) {
                index.push_back(j);
                transpose(i + 1, new_tensor, trans, index);
                index.pop_back();
            }
        }

        void update_strided();

        bool shapeEq(const _Tensor& t) const;

        std::vector<int> _shape = {};
        std::vector<float> _data = {};
        std::vector<int> _strided = {};
        
        int _size = 0;

        mutable std::shared_ptr<_Tensor> operand1 = nullptr;
        mutable std::shared_ptr<_Tensor> operand2 = nullptr;
        
        mutable std::vector<float> _grad = {};

        mutable std::shared_ptr<_Tensor> bias = nullptr;

        mutable float const_operand1 = 0.f;
        mutable float const_operand2 = 0.f;

        mutable int opeartor = -1;

        mutable bool _require_grad = true;
    };

    class Tensor {
    public:
        Tensor() {
            _tensor = std::make_shared<_Tensor>();
        }

        Tensor(const Tensor& t) {
            _tensor = t._tensor;
        }

        Tensor(_Tensor& t) {
            _tensor = std::shared_ptr<_Tensor>(&t);
        }

        Tensor(Tensor&& t)  noexcept {
            _tensor = std::move(t._tensor);
        }

        Tensor(const std::vector<int> shape) {
            _tensor = std::make_shared<_Tensor>(shape);
        }

        template<typename T, typename... Args> requires (std::is_same_v<T, int> || std::is_same_v<T, size_t>)
        Tensor(const T& arg, const Args&... args) {
            _tensor = std::make_shared<_Tensor>(_Tensor(arg, args...));
        }

        ~Tensor() {
        }

        template<typename... Args>
        Tensor reshape(const Args&... args) {
            return Tensor(_tensor->reshape(args...));
        }

        template<typename... Args>
        float& at(const Args&... args) {
            return _tensor->at(args...);
        }

        template<typename... Args>
        float at(const Args&... args) const {
            return _tensor->at(args...);

        }

        template<typename T, typename... Args> requires (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>)
        void transpose(const T& arg, const Args&... args) {
            _tensor->transpose(arg, args...);
        }

        float& at(const std::vector<int>& indexs) {
            return _tensor->at(indexs);
        }

        float at(const std::vector<int>& indexs) const {
            return _tensor->at(indexs);
        }

        const std::vector<int> shape() const {
            return _tensor->shape();
        }
        int size() const {
            return _tensor->size();
        }

        std::vector<float>::iterator begin() {
            return _tensor->begin();
        }
        std::vector<float>::iterator end() {
            return _tensor->end();
        }

        std::vector<float>::iterator begin() const {
            return _tensor->begin();
        }
        std::vector<float>::iterator end() const {
            return _tensor->end();
        }

        std::vector<float>::iterator grad_begin() {
            return _tensor->grad_begin();
        }
        std::vector<float>::iterator grad_end() {
            return _tensor->grad_end();
        }

        std::vector<float>::iterator bias_begin() {
            return _tensor->bias_begin();
        }
        std::vector<float>::iterator bias_end() {
            return _tensor->bias_end();
        }
        bool bias_size() {
            return (_tensor->bias_size());
        }

        void set_conv_bias(int num) {
            _tensor->set_conv_bias(num);
        }

        std::vector<float>::iterator bias_grad_begin() {
            return _tensor->bias_grad_begin();
        }
        std::vector<float>::iterator bias_grad_end() {
            return _tensor->bias_grad_end();
        }

        void assignment(float other) {
            _tensor->assignment(other);
        }

        void rand() {
            _tensor->rand();
        }

        float max() const {
            return _tensor->max();
        }

        Tensor operator=(const Tensor& t) {
            _tensor = t._tensor;
            return *this;
        }
        Tensor operator=(Tensor&& t) noexcept {
            _tensor = std::move(t._tensor);
            return *this;
        }
        Tensor operator=(float other) {
            *_tensor = other;
            return *this;
        }

        friend Tensor operator+(const Tensor& t, float other) {
            return Tensor(*(t._tensor) + other);
        }
        friend Tensor operator+(float other, const Tensor& t) { 
            return Tensor(*(t._tensor) + other);
        }

        friend Tensor operator-(const Tensor& t, float other) {
            return Tensor(*(t._tensor) - other);
        }
        friend Tensor operator-(float other, const Tensor& t) {
            return Tensor(other - *(t._tensor));
        }

        friend Tensor operator*(const Tensor& t, float other) {
            return Tensor(*(t._tensor) * other);
        }
        friend Tensor operator*(float other, const Tensor& t) {
            return Tensor(*(t._tensor) * other);
        }

        friend Tensor operator/(const Tensor& t, float other) {
            return Tensor(*(t._tensor) / other);
        }
        friend Tensor operator/(float other, const Tensor& t) {
            return Tensor(other / *(t._tensor));
        }

        Tensor operator+(const Tensor& t) const {
            return Tensor(*(_tensor) + *(t._tensor));
        }
        Tensor operator-(const Tensor& t) const {
            return Tensor(*(_tensor) - *(t._tensor));
        }
        Tensor operator*(const Tensor& t) const {
            return Tensor(*(_tensor) * *(t._tensor));
        }
        Tensor operator/(const Tensor& t) const {
            return Tensor(*(_tensor) / *(t._tensor));
        }

        Tensor operator+=(float other) {
            _tensor = std::shared_ptr<_Tensor>(&(*(_tensor)*other));
            return *this;
        }
        Tensor operator-=(float other) {
            _tensor = std::shared_ptr<_Tensor>(&(*(_tensor)*other));
            return *this;
        }
        Tensor operator*=(float other) {
            _tensor = std::shared_ptr<_Tensor>(&(*(_tensor)*other));
            return *this;
        }
        Tensor operator/=(float other) {
            _tensor = std::shared_ptr<_Tensor>(&(*(_tensor) * other));
            return *this;
        }
        Tensor operator+=(const Tensor& t) {
            _tensor = std::shared_ptr<_Tensor>(&(*(_tensor) + *(t._tensor)));
            return *this;
        }
        Tensor operator-=(const Tensor& t) {
            _tensor = std::shared_ptr<_Tensor>(&(*(_tensor) - *(t._tensor)));
            return *this;
        }
        Tensor operator*=(const Tensor& t) {
            _tensor = std::shared_ptr<_Tensor>(&(*(_tensor) * *(t._tensor)));
            return *this;
        }
        Tensor operator/=(const Tensor& t) {
            _tensor = std::shared_ptr<_Tensor>(&(*(_tensor) / *(t._tensor)));
            return *this;
        }

        Tensor operator-() const {
            return Tensor(-*(_tensor));
        }

        bool operator==(const Tensor& t) const {
            return *(_tensor) == *(t._tensor);
        }
        bool operator!=(const Tensor& t) const {
            return *(_tensor) != *(t._tensor);
        }

        friend std::ostream& operator<<(std::ostream& out, const Tensor& tensor) {
            out << *(tensor._tensor);
            return out;
        }

        void from_vector_1d(const std::vector<float>& v, int s = 0, int e = -1) {
            _tensor->from_vector_1d(v, s, e);
        }
        void from_vector_2d(const std::vector<std::vector<float>>& v, int s = 0, int e = -1) {
            _tensor->from_vector_2d(v, s, e);
        }
        Tensor dropout(float proportion) const {
            return Tensor(_tensor->dropout(proportion));
        }
        Tensor maxpool2d(size_t size = 2) const {
            return Tensor(_tensor->maxpool2d(size));
        }
        Tensor conv2d(const Tensor& other,int padding, int stride_x = 1, int stride_y = 1) const {
            return Tensor(padding2d(padding)._tensor->conv2d(*(other._tensor), stride_x, stride_y));
        }
        Tensor padding2d(int size) const {
            return Tensor(_tensor->padding2d(size));
        }
        Tensor mean() const {
            return Tensor(_tensor->mean());
        }
        Tensor dot2d(const Tensor& other) {
            return Tensor(_tensor->dot2d(*(other._tensor)));
        }
        Tensor pow(float s) const {
            return Tensor(_tensor->pow(s));
        }
        Tensor rpow(float s) const {
            return Tensor(_tensor->rpow(s));
        }
        Tensor log(float s = E) const {
            return Tensor(_tensor->log(s));
        }
        Tensor sum() const {
            return Tensor(_tensor->sum());
        }
        Tensor relu() const {
            return Tensor(_tensor->relu());
        }
        Tensor rowsum() const {
            return Tensor(_tensor->rowsum());
        }
        Tensor colmean() const {
            return Tensor(_tensor->colmean());
        }

        void require_grad(bool grad) const {
            _tensor->require_grad(grad);
        }
        bool require_grad() const {
            return _tensor->require_grad();
        }

        void zero_grad() {
            _tensor->zero_grad();
        }
        void backward() {
            _tensor->backward();
        }
    private:
        std::shared_ptr<_Tensor> _tensor;
    };
} // namespace minnet

#endif
