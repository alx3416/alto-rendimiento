#include "image_utils.h"


int main(int, char**)
{
    cv::Mat frame, frame_gray, frame_gray_gamma;
    cv::Mat image = cv::imread("baboon.png");
    if (image.empty())
    {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    cv::cvtColor(image, frame_gray, cv::COLOR_BGR2GRAY);
    frame_gray_gamma = frame_gray.clone();
    myLib::gammaCorrectionPassByReference(frame_gray, frame_gray_gamma, 0.25);

    cv::imshow("Image_gray", frame_gray);
    cv::imshow("Image_gamma", frame_gray_gamma);
    cv::waitKey(0); // Wait for a key press
    return 0;
}
