#ifndef CPPBASICS_IMAGE_UTILS_H
#define CPPBASICS_IMAGE_UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>

namespace myLib{

    cv::Mat gammaCorrectionPassByValue(cv::Mat frame_gray){
        double value = 0;
        for(int row=0; row<frame_gray.rows; row++){
            for(int col=0; col<frame_gray.cols; col++){
                value = static_cast<double>(frame_gray.at<uint8_t >(row, col));

                // TODO Your code here

                frame_gray.at<uint8_t >(row, col) = static_cast<uint8_t>(value);
            }
        }
        return frame_gray;
    }

// TODO: code the same gamma correction function but now with pass by reference

// TODO: code two functions, one for kernel and local patch multiplication, and another one to perform sliding window

} // namespace myLib

#endif //CPPBASICS_IMAGE_UTILS_H
