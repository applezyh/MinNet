#include <iostream>
#include <vector>

#define E 2.718281828459045235360287471352662497757247093f

namespace minnet
{
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
        RELU
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

        template<typename... Args>
        _Tensor(const Args&... args) :_size(1) {
            Init(args...);
            _data = std::vector<float>(_size, 0.f);
            _grad = std::vector<float>(_size, 0.f);
            update_strided();
        }

        template<typename T, typename... Args>  requires (std::is_same_v<T, int>)
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

        template<typename T, typename... Args>  requires (std::is_same_v<T, int>)
        void reshape(const T& arg, const Args&... args) {
            int temp = _size;
            std::vector<int> shape = _shape;
            _size = 1;
            _shape.clear();
            Init(arg, args...);
            if (temp == _size) {
                update_strided();
                return;
            }
            else {
                int count = 0;
                int temp_size = 1;
                for (int i = 0; i < _shape.size(); i++) {
                    if (_shape[i] == -1) count++;
                    else temp_size *= _shape[i];
                }
                if (temp_size != 0 && count == 1 && temp % temp_size == 0) {
                    int t = temp / temp_size;
                    for (int i = 0; i < _shape.size(); i++) {
                        if (_shape[i] == -1) _shape[i] = t;
                    }
                    update_strided();
                    return;
                }
                std::vector<int> wrong_shape = _shape;
                _size = temp;
                _shape = shape;
                throw TensorWrong(2, _shape, wrong_shape);
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

        template<typename T, typename... Args> requires (std::is_same_v<T, int>)
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

        template<typename T, typename... Args> requires (std::is_same_v<T, int>)
        void getIndex(int& index, int& i, const T& arg, const Args&... args) const {
            index += arg * _strided[i++];
            getIndex(index, i, args...);
        }

        void getIndex(int& index, int& i) const {
            return;
        }

        template<typename T, typename... Args> requires (std::is_same_v<T, int>)
            void getIndex(std::vector<int>& index, const T& arg, const Args&... args) const {
            index.push_back(arg);
            getIndex(index, args...);
        }

        template<typename T> requires (std::is_same_v<T, int>)
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

        template<typename T, typename... Args> requires (std::is_same_v<T, int>)
        Tensor(const T& arg, const Args&... args) {
            _tensor = std::make_shared<_Tensor>(_Tensor(arg, args...));
        }

        ~Tensor() {
        }

        template<typename... Args>
        void reshape(const Args&... args) {
            return _tensor->reshape(args...);
        }

        template<typename... Args>
        float& at(const Args&... args) {
            return _tensor->at(args...);
        }

        template<typename... Args>
        float at(const Args&... args) const {
            return _tensor->at(args);

        }

        template<typename T, typename... Args> requires (std::is_same_v<T, int>)
        void transpose(const T& arg, const Args&... args) {
            _tensor->transpose(arg, args);
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

        std::vector<float>::iterator grad_begin() {
            return _tensor->grad_begin();
        }
        std::vector<float>::iterator grad_end() {
            return _tensor->grad_end();
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
