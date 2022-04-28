#include "transformation.hpp"

namespace minnet {
    Tensor from_mat(const cv::Mat& img) {
        if (img.channels() == 1) img.convertTo(img, CV_32F);
        else if (img.channels() == 3) img.convertTo(img, CV_32FC3);
        else return Tensor();
        cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);
        Tensor ret(img.rows, img.cols, img.channels());
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                for (int k = 0; k < img.channels(); k++) {
                    ret.at(i, j, k) = img.ptr<float>(i, j)[k];
                }
            }
        }
        return ret;
    }

    cv::Mat to_mat(const Tensor& tensor) {
        cv::Mat img;
        if (tensor.shape().size() != 3
            || (tensor.shape()[2] != 1 && tensor.shape()[2] != 3)) return img;
        if (tensor.shape()[2] == 1) img.create(tensor.shape()[0], tensor.shape()[1], CV_32F);
        else if (tensor.shape()[2] == 3) img.create(tensor.shape()[0], tensor.shape()[1], CV_32FC3);
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                for (int k = 0; k < img.channels(); k++) {
                    img.ptr<float>(i, j)[k] = tensor.at(i, j, k);
                }
            }
        }
        return img;
    }
} // minnet