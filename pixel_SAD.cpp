#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include "image.h"

struct Data {
	Image<Vec3b> I1, I2; // original rgb images
	Image<uchar> G1, G2; // grayscale versions of rgb images
	Image<float> F1, F2; // floating point version of rgb images
	Matx33d F; // Fundamental matrix
};

void calculate_depth_map(Data D){
    cv::Mat matches(D.I1.rows, D.I1.cols, CV_32SC1);
    for(int i=0;i<D.I1.rows;i++){
    	for(int j=0;j<D.I1.cols;j++)
    		matches.at<int>(i,j) = -1; // for mismatches
    }
    int m = D.I1.rows;
    int n = D.I1.cols;

    // the F matrix for car
    D.F(0,0) = 7.7423681e-07; D.F(0,1) = 7.9756355e-05; D.F(0,2) = -0.016849758;
    D.F(1,0) = -7.3229807e-05; D.F(1,1) = 2.5419604e-06; D.F(1,2) =  0.02944893;
    D.F(2,0) = 0.014332652; D.F(2,1) = -0.033120934; D.F(2,2) =  0.99877244;
    
    // the F matrix for tsukuba
    // D.F(0,0) = 8.9194209e-06; D.F(0,1) = -0.00021785361; D.F(0,2) = 0.017634209;
    // D.F(1,0) = 0.00023563496; D.F(1,1) = 1.3251083e-05; D.F(1,2) = -0.039607145;
    // D.F(2,0) = -0.018771617; D.F(2,1) = 0.023556622; D.F(2,2) = 0.99860555; 

	// Comparison of pixels
	for(int i=0;i<D.I1.rows;i++){
		for(int j=0;j<D.I1.cols;j++){
			Point p1 (i,j);
			Vec3d m1 (p1.y, p1.x, 1);
			Vec3d l = D.F*m1; // Epipolar line equation
	    	double dissimilarity = 1000000.;
			double best_corr = -1; // worst case
			for(int k=0;k<D.I2.rows;k++){
				int y = k;
				int x = -(y*l(1)+l(2))/l(0);

				if (x < 0 || x > D.I2.cols-1){
					continue;
				}

				Point p2(y,x);
                double cur_dissimilarity = abs(D.I1.at<Vec3b>(p1.x,p1.y)[0]-D.I2.at<Vec3b>(p2.x,p2.y)[0])
                						 + abs(D.I1.at<Vec3b>(p1.x,p1.y)[1]-D.I2.at<Vec3b>(p2.x,p2.y)[1])
                						 + abs(D.I1.at<Vec3b>(p1.x,p1.y)[2]-D.I2.at<Vec3b>(p2.x,p2.y)[2]);

                if(cur_dissimilarity < 250){
                	if(cur_dissimilarity < dissimilarity){
                   		matches.at<int>(i,j) = x;
                    	dissimilarity = cur_dissimilarity;
                	}
                }
			}
		}
	}
	// std::cout << "Finished dissimilarity calculation.\n";


	// Dissimilarity summary report
	// int actual_matches = 0, nb_no_shifts = 0;
	// for(int i=0;i<matches.rows;i++)
	// 	for(int j=0;j<matches.cols;j++){
	// 		if ( j == matches.at<int>(i,j) )
	// 			nb_no_shifts++;
	// 		if(matches.at<int>(i,j) != -1){
	// 			// std::cout << "Left (" << i << ", " << j << "), " << "Right (" << matches.at<int>(i,j) << ").\n";
	// 			actual_matches++;
	// 		}
	// 	}
	// std::cout << "number of actual matches is: " << actual_matches << " out of "
	// << D.I1.cols*D.I1.rows << ", " << (float)actual_matches/(D.I1.cols*D.I1.rows)*100 << "%\n";
	// std::cout << "number of No shift matches is: " << nb_no_shifts << " out of "
	// << D.I1.cols*D.I1.rows << ", " << (float)nb_no_shifts/(D.I1.cols*D.I1.rows)*100 << "%\n";


	// Disparity matrix
	Mat disparity(m,n,CV_8UC1);
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++)
			disparity.at<uchar>(i,j) = abs(j-matches.at<int>(i,j));
	}

	// Depth calculation
	Mat Z(m, n, CV_32F); // depth for all pixels
	double f = 26; // focal length of the camera = 26mm (52mm)
	double d = 100; // the distance between focuses of two cameras = 10cm
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
				// if(Z.at<float>(i,j) != -1)
				if(j == matches.at<int>(i,j)){ // No shift: important to avoid inf, but better?
					// Z.at<float>(i,j) = (f*d)/abs(j + 1 - matches.at<int>(i,j));
					continue;
				}
				else
					// disparity - in pixels, f in pixels, the others in millimeters.
					Z.at<float>(i,j) = (f*d)/abs(j - matches.at<int>(i,j));
		}
	}
	// std::cout << "Finished depth calculation.\n";

    // Maximum depth calculation
    double mindepth, maxdepth;
    cv::Mat depth(m, n, CV_8UC1);
    cv::minMaxLoc(Z, &mindepth, &maxdepth);

    // calculation of pixel value
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++){
			if (matches.at<int>(i,j) == -1) // no match
	        	depth.at<uchar>(i,j) = 255;
	        else if (Z.at<float>(i,j) == maxdepth) // the pixels with maxdepth
	        	depth.at<uchar>(i,j) = 0;
   			else{
   				if(j != matches.at<int>(i,j)) // for no shift condition
		            depth.at<uchar>(i,j) = 255 - int((Z.at<float>(i,j) * 255.)/double(maxdepth));
		       	else
		       		depth.at<uchar>(i,j) = 255;
   			}
        }

    // swithcing pixels
    for(int i=0;i<m;i++){
    	for(int j=0;j<n;j++){
    		depth.at<uchar>(i,j) = 255 - depth.at<uchar>(i,j);
    	}
    }

    // final steps
    cv::imwrite("../runs/disparity/disparity2.jpg", disparity);
    // cv::imwrite("../runs/disparity/depth_map2.jpg", depth);

   	cv::imshow("disparity", disparity);
    // cv::imshow("Depth map", depth);
}

int main(int argc, char** argv)
{
	Data D;
	D.I1 = imread("../runs/detect/exp/car_blue_1.jpeg_cropped_1.jpg");
	D.I2 = imread("../runs/detect/exp/car_blue_2.jpeg_cropped_1.jpg");
	// imshow("I1", D.I1);
	// imshow("I2", D.I2);

	Image<uchar>G1, G2;
	cvtColor(D.I1, D.G1, COLOR_BGR2GRAY);
	cvtColor(D.I2, D.G2, COLOR_BGR2GRAY);
	D.G1.convertTo(D.F1, CV_32F);
	D.G2.convertTo(D.F2, CV_32F);

	calculate_depth_map(D);
	waitKey(0);

	return 0;
}
