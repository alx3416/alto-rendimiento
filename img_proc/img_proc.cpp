#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <cmath>


Eigen::MatrixXd cvMatToEigen(const cv::Mat& mat) {
    Eigen::MatrixXd eigen_mat(mat.rows, mat.cols);
    cv::cv2eigen(mat, eigen_mat);
    return eigen_mat;
}

int main(int, char**)
{
    cv::Mat frame, frame_gray, frame_gray_processed;
    cv::Mat image = cv::imread("baboon.png");
    if (image.empty())
    {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    cv::cvtColor(image, frame_gray, cv::COLOR_BGR2GRAY);
    frame_gray_processed = frame_gray.clone();
    cv::Mat imagen_double;
    frame_gray.convertTo(imagen_double, CV_64F);
    Eigen::MatrixXd eigen_image = cvMatToEigen(imagen_double);
    std::cout << "el valor del pixel es: " << eigen_image(128,128) << std::endl;


    cv::imshow("Image_gray", frame_gray);
    cv::imshow("Image_processed", frame_gray_processed);
    cv::waitKey(0); // Wait for a key press
    return 0;
}