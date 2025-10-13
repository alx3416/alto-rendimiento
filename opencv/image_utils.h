#ifndef CPPBASICS_IMAGE_UTILS_H
#define CPPBASICS_IMAGE_UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>

namespace myLib
{
    cv::Mat gammaCorrectionPassByValue(cv::Mat frame_gray, const float gamma)
    {
        cv::Mat result = frame_gray.clone();
        double value = 0;
        for (int row = 0; row < frame_gray.rows; row++)
        {
            for (int col = 0; col < frame_gray.cols; col++)
            {
                value = static_cast<double>(frame_gray.at<uint8_t>(row, col));

                value = value / 255.0;
                value = std::pow(value, gamma);
                value = value * 255.0;

                result.at<uint8_t>(row, col) = static_cast<uint8_t>(value);
            }
        }
        return result;
    }

    void gammaCorrectionPassByReference(cv::Mat& source_gray, cv::Mat& result_gray, const float gamma)
    {
        double value = 0;
        for (int row = 0; row < source_gray.rows; row++)
        {
            for (int col = 0; col < source_gray.cols; col++)
            {
                value = static_cast<double>(source_gray.at<uint8_t>(row, col));

                value = value / 255.0;
                value = std::pow(value, gamma);
                value = value * 255.0;

                result_gray.at<uint8_t>(row, col) = static_cast<uint8_t>(value);
            }
        }
    }

    // TODO: code two functions, one for kernel and local patch multiplication, and another one to perform sliding window
} // namespace myLib

#endif //CPPBASICS_IMAGE_UTILS_H
