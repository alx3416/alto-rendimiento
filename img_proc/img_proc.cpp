#include "img_proc.hpp"
#include <chrono>

int main(int, char **) {
    omp_set_num_threads(4);
    cv::Mat frame, frame_gray, frame_gray_processed, frame_gray_processed_2;
    cv::Mat image = cv::imread("baboon.png");
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    cv::cvtColor(image, frame_gray, cv::COLOR_BGR2GRAY);
    frame_gray_processed = frame_gray.clone();
    cv::Mat imagen_double;
    frame_gray.convertTo(imagen_double, CV_64F);
    Eigen::MatrixXd eigen_image = EigenCV::cvMatToEigen(imagen_double / 255);

    // Define kernel
    Eigen::MatrixXd kernel(3, 3);
    kernel(0, 0) = -1;
    kernel(0, 1) = 0;
    kernel(0, 2) = 1;
    kernel(1, 0) = -2;
    kernel(1, 1) = 0;
    kernel(1, 2) = 2;
    kernel(2, 0) = -1;
    kernel(2, 1) = 0;
    kernel(2, 2) = 1;

    std::cout << "Kernel:\n" << kernel << std::endl;
    std::cout << "Pixel value at (128,128): " << eigen_image(128, 128) << std::endl;
    std::cout << "\nPerforming warm-up runs..." << std::endl;

    // Warm-up runs (to avoid cold cache effects)
    Eigen::MatrixXd eigen_result = EigenCV::eigen_convolution(eigen_image, kernel);
    Eigen::VectorXd vector1(3);
    vector1 << -1, 0, 1;
    Eigen::VectorXd vector2(3);
    vector2 << 1, 2, 1;
    Eigen::MatrixXd eigen_result_2 = EigenCV::eigen_separable_convolution(eigen_image, vector1, vector2);

    // Time with multiple iterations
    const int num_iterations = 100;
    std::cout << "Running " << num_iterations << " iterations for each method..." << std::endl;

    // Time normal 2D convolution
    auto start_normal = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        eigen_result = EigenCV::eigen_convolution(eigen_image, kernel);
    }
    auto end_normal = std::chrono::high_resolution_clock::now();

    // Time separable convolution
    auto start_separable = std::chrono::high_resolution_clock::now();
    // Perform SVD on the kernel
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(kernel, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // For a rank-1 separable kernel, use the first singular value
    // and corresponding singular vectors
    double sigma = svd.singularValues()(0);
    Eigen::VectorXd u = svd.matrixU().col(0) * std::sqrt(sigma);
    Eigen::VectorXd v = svd.matrixV().col(0) * std::sqrt(sigma);
    for (int i = 0; i < num_iterations; i++) {
        eigen_result_2 = EigenCV::eigen_separable_convolution(eigen_image, u, v);
    }
    auto end_separable = std::chrono::high_resolution_clock::now();

    // Calculate average durations
    auto total_duration_normal = std::chrono::duration_cast<std::chrono::microseconds>(end_normal - start_normal).
            count();
    auto total_duration_separable = std::chrono::duration_cast<std::chrono::microseconds>(
        end_separable - start_separable).count();

    double avg_duration_normal = (double) total_duration_normal / num_iterations;
    double avg_duration_separable = (double) total_duration_separable / num_iterations;

    frame_gray_processed = EigenCV::eigenTocvMat(eigen_result);
    frame_gray_processed_2 = EigenCV::eigenTocvMat(eigen_result_2);

    std::cout << "\n=== Timing Results ===" << std::endl;
    std::cout << "Normal 2D convolution:" << std::endl;
    std::cout << "  Total time: " << total_duration_normal << " microseconds" << std::endl;
    std::cout << "  Average per iteration: " << avg_duration_normal << " microseconds" << std::endl;
    std::cout << "\nSeparable convolution:" << std::endl;
    std::cout << "  Total time: " << total_duration_separable << " microseconds" << std::endl;
    std::cout << "  Average per iteration: " << avg_duration_separable << " microseconds" << std::endl;
    std::cout << "\nSpeedup: " << avg_duration_normal / avg_duration_separable << "x" << std::endl;

    cv::imshow("Image_gray", frame_gray);
    cv::imshow("Image_processed", frame_gray_processed);
    cv::imshow("Image_processed_separable", frame_gray_processed_2);
    cv::waitKey(0);
    return 0;
}
