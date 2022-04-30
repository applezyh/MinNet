#include "Dataset.hpp"

std::vector<float> image_to_vec(cv::Mat& m) {
    std::vector<float> result(m.rows * m.cols * m.channels());
    int t = 0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            for (int k = 0; k < m.channels(); k++) {
                result[t++] = m.ptr<uchar>(i, j)[k] / 255.f;
            }
        }
    }
    return result;
}
