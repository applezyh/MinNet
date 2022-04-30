#ifndef CIFA10_H_
#define CIFA10_H_
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

std::vector<std::pair<int, cv::Mat>> load_cifa10(const std::string& cifa10_img_dir);


#endif // !CIFA10_H_

