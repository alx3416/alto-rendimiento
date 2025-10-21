#ifndef ALTO_RENDIMIENTO_IMG_PROC_HPP
#define ALTO_RENDIMIENTO_IMG_PROC_HPP

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <cmath>

namespace EigenCV {
    Eigen::MatrixXd cvMatToEigen(const cv::Mat &mat) {
        Eigen::MatrixXd eigen_mat(mat.rows, mat.cols);
        cv::cv2eigen(mat, eigen_mat);
        return eigen_mat;
    }

    Eigen::MatrixXd eigen_convolution(const Eigen::MatrixXd &image, const Eigen::MatrixXd &kernel) {
        Eigen::MatrixXd res = image;
        auto patch = kernel;
        for (int row = 1; row < (image.rows() - 1); row++) {
            for (int col = 1; col < (image.cols() - 1); col++) {

                patch = image.block(row-1, col-1, kernel.rows(), kernel.cols());
                patch = patch.cwiseProduct(kernel);
                res(row, col) = patch.sum();

            }
        }
        return res;
    }

    cv::Mat eigenTocvMat(const Eigen::MatrixXd &eigen_data) {
        cv::Mat result(eigen_data.rows(), eigen_data.cols(), CV_8U);
        eigen_data.cwiseAbs() * 255;
        cv::eigen2cv(eigen_data, result);
        return result;
    }

}

#endif //ALTO_RENDIMIENTO_IMG_PROC_HPP
