#ifndef TRANS_H_
#define TRANS_H_
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Tensor.hpp"

namespace minnet {

    Tensor from_mat(const cv::Mat& img);

    cv::Mat to_mat(const Tensor& tensor);
} // minnet
#endif