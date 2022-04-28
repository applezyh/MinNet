#ifndef MNIST_H_
#define MNIST_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

std::vector<std::pair<int, cv::Mat>> readAndSave(const std::string& mnist_img_path, const std::string& mnist_label_path);

std::vector<float> image_to_vec(cv::Mat& m);

#endif