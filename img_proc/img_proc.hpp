#ifndef ALTO_RENDIMIENTO_IMG_PROC_HPP
#define ALTO_RENDIMIENTO_IMG_PROC_HPP

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <cmath>
#include <omp.h>

namespace EigenCV {
    Eigen::MatrixXd cvMatToEigen(const cv::Mat &mat) {
        Eigen::MatrixXd eigen_mat(mat.rows, mat.cols);
        cv::cv2eigen(mat, eigen_mat);
        return eigen_mat;
    }

    Eigen::MatrixXd eigen_convolution(const Eigen::MatrixXd &image, const Eigen::MatrixXd &kernel) {
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(image.rows(), image.cols());
#pragma omp parallel for
        for (int row = 1; row < (image.rows() - 1); row++) {
            for (int col = 1; col < (image.cols() - 1); col++) {
                res(row, col) = (image.block(row - 1, col - 1, kernel.rows(), kernel.cols()).cwiseProduct(kernel)).
                        sum();
            }
        }
        return res;
    }

    // 1D convolution along rows
    Eigen::MatrixXd convolve_rows(const Eigen::MatrixXd &image, const Eigen::VectorXd &kernel) {
        int kernel_radius = kernel.size() / 2;
        Eigen::MatrixXd res(image.rows(), image.cols());
        res.setZero(); // Initialize to zero
#pragma omp parallel for
        for (int row = 0; row < image.rows(); row++) {
            for (int col = kernel_radius; col < (image.cols() - kernel_radius); col++) {
                double sum = 0.0;
                for (int k = 0; k < kernel.size(); k++) {
                    sum += image(row, col - kernel_radius + k) * kernel(k);
                }
                res(row, col) = sum;
            }
        }
        return res;
    }

    // 1D convolution along columns
    Eigen::MatrixXd convolve_cols(const Eigen::MatrixXd &image, const Eigen::VectorXd &kernel) {
        int kernel_radius = kernel.size() / 2;
        Eigen::MatrixXd res = image;
#pragma omp parallel for
        for (int row = kernel_radius; row < (image.rows() - kernel_radius); row++) {
            for (int col = 0; col < image.cols(); col++) {
                double sum = 0.0;
                for (int k = 0; k < kernel.size(); k++) {
                    sum += image(row - kernel_radius + k, col) * kernel(k);
                }
                res(row, col) = sum;
            }
        }
        return res;
    }

    Eigen::MatrixXd eigen_separable_convolution(const Eigen::MatrixXd &image, const Eigen::VectorXd &u, const Eigen::VectorXd &v) {

        // Apply separable convolution: first along columns (with u), then along rows (with v)
        Eigen::MatrixXd temp = convolve_cols(image, u);
        Eigen::MatrixXd res = convolve_rows(temp, v);

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
