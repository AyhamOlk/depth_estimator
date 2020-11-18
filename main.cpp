#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
// #include "image.h"

using namespace cv;

int main(int argc, char** argv)
{
	cv::Mat img1 = cv::imread("../data/sun0.h", IMREAD_UNCHANGED);
    cv::Mat img2 = cv::imread("../data/demo.");


    // cv::imshow("Img1", img1);
    // std::cout << img1.channels() << std::endl;
    // std::cout << img1.size() << std::endl;

    // TODO: for now images are of the same size.
    // for(int i=0;i<img1.rows)


    // Depth map algorithm
    


	// G1.convertTo(D.F1, CV_32F);
	// G2.convertTo(D.F2, CV_32F);

	// setMouseCallback("I1", onMouse1, &D);
	// setMouseCallback("I2", onMouse2, &D);

	cv::waitKey(0);
	return 0;
}
