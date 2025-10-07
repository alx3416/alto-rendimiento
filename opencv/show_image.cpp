#include "image_utils.h"


int main(int, char**)
{
    cv::Mat frame, frame_gray;
    cv::Mat image = cv::imread("baboon.png");
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    cv::imshow("Image", image);
    cv::waitKey(0); // Wait for a key press
    return 0;
}