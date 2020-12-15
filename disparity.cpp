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

void onMouse1(int event, int x, int y, int foo, void* p)
{
	if (event != EVENT_LBUTTONDOWN)
		return;
	Point m1(x, y);

	Data* D = (Data*)p;
	circle(D->I1, m1, 2, Scalar(0, 255, 0), 2);
	imshow("I1", D->I1);

	Vec3d m1p(m1.x, m1.y, 1);
	Vec3d l = D->F*m1p; // Epipolar line equation

	Point m2a(0,-l(2)/(l(1))), m2b(D->I2.width(),-(D->I2.width()*l(0)+l(2))/l(1));
	line(D->I2,m2a,m2b,Scalar(0,255,0),1);

	Point m2, best_p;
	double best_corr = -1; // worst case
	for(int i=0;i<D->F2.width();i++){
		int x = i;
		int y = -(x*l(0)+l(2))/l(1);
		m2.x = x; m2.y = y;
		double cur_corr = NCC(D->F1,m1,D->F2,m2,25);
		if (cur_corr > best_corr){
			best_corr = cur_corr;
			best_p.x = x; best_p.y = y;
		}
	}
	circle(D->I2, best_p, 2, Scalar(0,255,0), 2);

	imshow("I2", D->I2);
}


double mean(Image<uchar> window){
	double mean = 0.;
	for(int i=0;i<window.rows;i++){
		for(int j=0;j<window.cols;j++){
			mean += double(window(i,j));
		}
	}
	return mean/(pow(window.rows, 2));
}

double std_dev(Image<uchar> window){
	double sigma = 0.;
	double average = mean(window);
	for(int i=0;i<window.rows;i++){
		for(int j=0;j<window.cols;j++){
			sigma += pow(double(window(i,j))-average, 2);
		}
	}

	return sqrt(sigma/pow(window.rows, 2));
}


double ZNCC (Image<uchar> window1, Image<uchar> window2){
	double sum = 0.;
	double mean1 = mean(window1), mean2 = mean(window2);
	double std1 = std_dev(window1), std2 = std_dev(window2);
	// std::cout << "Mean 1=" << mean1 << ", Mean2 ="  << mean2 << '\n';
	// std::cout << "std 1=" << std1 << ", Std2 ="  << std2 << '\n';

	for(int i=0;i<window1.rows;i++){
		for(int j=0;j<window1.cols;j++){
			sum += (window1(i,j) - mean1)*(window2(i,j) - mean2);
		}
	}

	double zncc = sum/(pow(window1.rows,2)*std1*std2 + 0.00001);
	return abs(zncc);
}

double SAD (Image<Vec3b>& I1, Image<Vec3b>& I2, int y1, int x1, int y2, int x2){
	if(y1 > I1.rows || x1 > I1.cols || y2 > I2.rows || x2 > I2.cols)
		return 0;

	double dissimilarity = abs(I1.at<Vec3b>(y1,x1)[0]-I2.at<Vec3b>(y2,x2)[0])
            			 + abs(I1.at<Vec3b>(y1,x1)[1]-I2.at<Vec3b>(y2,x2)[1])
                		 + abs(I1.at<Vec3b>(y1,x1)[2]-I2.at<Vec3b>(y2,x2)[2]);
	return dissimilarity;
}

double SSD (Image<Vec3b>& I1, Image<Vec3b>& I2, int y1, int x1, int y2, int x2){
	double dissimilarity = pow(I1.at<Vec3b>(y1,x1)[0]-I2.at<Vec3b>(y2,x2)[0], 2)
            			 + pow(I1.at<Vec3b>(y1,x1)[1]-I2.at<Vec3b>(y2,x2)[1], 2)
                		 + pow(I1.at<Vec3b>(y1,x1)[2]-I2.at<Vec3b>(y2,x2)[2], 2);
	return dissimilarity;
}


// TODO: finish building window
std::vector<Point> build_epi_window(Image<uchar>& right, Image<uchar>& I2, Vec3d epipolar_line, int x, int y){
	std::vector<Point> best_points; // initialize with 0s?
	if(x + right.cols > I2.cols-2 || y + right.rows > I2.rows-2){
		return best_points;
	}
	int cur_x = x, cur_y = y;

	Vec3d perp_line(3,1,CV_32F);
	perp_line(0) = -epipolar_line(1);
	perp_line(1) = epipolar_line(0);
	perp_line(2) = -perp_line(0)*cur_x - perp_line(1)*cur_y;
	Point m2a(0, -1*(perp_line[0]*0 + perp_line[2])/perp_line[1]);
	Point m2b(I2.width(), -1*(perp_line[0]*I2.width() + perp_line[2])/perp_line[1]);
	// line(I2,m2a,m2b,Scalar(0,255,0),1);

	// imshow("Perp", I2);
	// waitKey(0);

	for(int r=0;r<right.rows;r++){
		for(int c=0;c<right.cols;c++){
			// right.at<uchar>(r,c) = 
			right(r,c) = I2(cur_x,cur_y);

			cur_x += 1;
			// epipolar_line(2) = epipolar_line(2) + r*epipolar_line(1);
			cur_y = -(cur_x*epipolar_line(0)+epipolar_line(2))/epipolar_line(1) + r;
			if(cur_y < 0 || cur_y > I2.rows-1){
				std::cout << "Current y = " << cur_y << ", " << "Error\n";
				continue;
			}
			best_points.push_back(Point(cur_x,cur_y));
		}
		// perp_line(2) = -perp_line(0)*cur_x - perp_line(1)*cur_y;
		// double val = (((perp_line(1)+perp_line(2))/perp_line(0))*epipolar_line(0) - epipolar_line(2))/epipolar_line(1);
		cur_x = x + r;
		// cur_y = val; //-(cur_x*perp_line(0)+perp_line(2))/perp_line(1);
		// cur_x = (-perp_line(1)*cur_y - perp_line(2))/perp_line(0);
		cur_y = -(cur_x*perp_line(0)+perp_line(2))/perp_line(1);

		std::cout << cur_x << ", " << cur_y << '\n';
	
		if(cur_x < 0 || cur_x > I2.cols-1 || cur_y < 0 || cur_y > I2.rows-1){
			// std::cout << "Error 2\n";
		}
	}
	return best_points;
}

void calculate_depth_map(Data D){
	int m = D.I1.rows, n = D.I1.cols;
	std::vector<Point> best_right_points;
	Point p_best;

	// TODO: do we need a matrix for matches?
    // cv::Mat matches(D.I1.rows, D.I1.cols, CV_32SC1, Scalar(255));
    cv::Mat matches(D.I1.rows, D.I1.cols, CV_32SC1);
    for(int i=0;i<D.I1.rows;i++){
    	for(int j=0;j<D.I1.cols;j++){
    		matches.at<int>(i,j) = -1; // for mismatches
    	}
    }

    // Fundamental matrix F
	// normalized non-resized F matrix for box flagged
	// D.F(0,0) = 1.67951e-07; D.F(0,1) = -8.05805e-09; D.F(0,2) = -1.16415e-10;
	// D.F(1,0) = 4.50064e-13; D.F(1,1) = -9.7441e-08; D.F(1,2) = -1.1205e-09;
    // D.F(2,0) = -0.000338589; D.F(2,1) = 0.000147331; D.F(2,2) = 1.72853e-06;

    // normalized F matrix for box
    D.F(0,0) = -6.5166212e-08; D.F(0,1) = -3.8590986e-08; D.F(0,2) = 0.00012766143;
    D.F(1,0) = -2.9300605e-08; D.F(1,1) = 2.7224738e-07; D.F(1,2) = 7.9271238e-05;
    D.F(2,0) = -7.4299576e-05; D.F(2,1) = -0.00040746361; D.F(2,2) = 0.11810446;

    // F matrix for box      
    // D.F(0,0) = -0.14897935; D.F(0,1) = -0.088224553; D.F(0,2) = -0.044179812;
    // D.F(1,0) = -0.066985406; D.F(1,1) = 0.62239677; D.F(1,2) = 0.67526948;
    // D.F(2,0) = -0.32830572; D.F(2,1) = -0.081912786; D.F(2,2) =  0.02076814;

    // F matrix for face
    // D.F(0,0) = 0.0004260085177640114; D.F(0,1) = 0.009695900301357771; D.F(0,2) = -7.287527222185076;
    // D.F(1,0) = -0.02048905307878512; D.F(1,1) = 0.003097332022218993; D.F(1,2) = 136.4338520488872;
    // D.F(2,0) = 8.778356935238595; D.F(2,1) = -135.7691005634687; D.F(2,2) = 2853.021715108659;

    // [!] F matrix unknown
    // D.F(0,0) = -1.0335373e-05; D.F(0,1) = 7.0145304e-12; D.F(0,2) = 0.0026458555;
    // D.F(1,0) = 3.6355606e-12; D.F(1,1) = -3.0976221e-06; D.F(1,2) = 0.00079298898;
    // D.F(2,0) = -1.1635342e-09; D.F(2,1) = 7.0720962e-10; D.F(2,2) = 1.1681963e-07;

    // F matrix for car
    // D.F(0,0) = 7.7423681e-07; D.F(0,1) = 7.9756355e-05; D.F(0,2) = -0.016849758;
    // D.F(1,0) = -7.3229807e-05; D.F(1,1) = 2.5419604e-06; D.F(1,2) =  0.02944893;
    // D.F(2,0) = 0.014332652; D.F(2,1) = -0.033120934; D.F(2,2) =  0.99877244;

    // F matrix for tsukuba
    // D.F(0,0) = 8.9194209e-06; D.F(0,1) = -0.00021785361; D.F(0,2) = 0.017634209;
    // D.F(1,0) = 0.00023563496; D.F(1,1) = 1.3251083e-05; D.F(1,2) = -0.039607145;
    // D.F(2,0) = -0.018771617; D.F(2,1) = 0.023556622; D.F(2,2) = 0.99860555;

    // Disparity calculation
	Mat disparity(m,n,CV_8UC1); // shift in position
	int width = 10, height = 10; // TODO: one size since the window patch is square?
	Image<uchar> left(width, height), right(width, height), best_right(width, height);

	// TODO: the condition to terminate
	for(int i=0;i<D.I1.rows-height;i+=height){
		for(int j=0;j<D.I1.cols-width;j+=width){
			// filling the windows with 0s
			for(int i=0;i<width;i++){
				for(int j=0;j<height;j++){
					right.at<uchar>(i,j) = 0;
					left.at<uchar>(i,j) = 0;
					best_right.at<uchar>(i,j)= 0;
				}
			}

			// creating left window
			for(int iLeft=i, c=0;iLeft<i+width;iLeft++, c++){
				for(int jLeft=j, d=0;jLeft<j+height;jLeft++, d++){
					Point p;
					p.y = iLeft; p.x = jLeft;
					left(c,d) = D.G1(iLeft,jLeft);
					circle(D.I1, p, 2, Scalar(0,255,0), 2);
				}
			}

			Point p1 (i,j);
			Vec3d m1 (p1.y, p1.x, 1);
			Vec3d l = D.F*m1; // Epipolar line equation
			Point m2a(0,-l(2)/(l(1))), m2b(D.I2.width(),-(D.I2.width()*l(0)+l(2))/l(1));
			// line(D.I2,m2a,m2b,Scalar(0,255,0),1);
			
			double best_corr = 0.; // worst case
			double dispar = 0.;
			for(int k=0;k<D.I2.cols;k++){
				int x = k;
				int y = -(x*l(0)+l(2))/l(1);

				// TODO: continue in the following condition or assign boundary values?
				if (y < 0 || y > D.I2.rows-1){
					// std::cout << "Continue\n";
					continue;
				}

				// creation of right window
				// best_right_points = build_epi_window(right, D.G2, l, x, y);
				
				Point p2(x,y);
				double cur_val = NCC(D.F1,p1,D.F2,p2,25);
				// std::cout << "P1 = " << p1 << ", P2 = " << p2 << ", cur_val = " << cur_val << '\n';
				// double cur_val = ZNCC(left, right);
				// double cur_val = SAD(D.I1, D.I2, i, j, y, x);
				if (cur_val > best_corr){
					// std::cout << "Val=" << cur_val << '\n';
					dispar = abs(y-i); // TODO: do we put the indices correctly?
					best_corr = cur_val;
					p_best = p2;
					best_right = right;
				}
			}
			// std::cout << p_best << '\n';
			best_right_points.push_back(p_best);
			circle(D.I2, p_best, 2, Scalar(0,255,0), 2);
			// std::cout << "Finished ZNCC\n";

			// fill disparity here after ZNCC (or any other metric) finishes
			// TODO: We are not considering other positions difference for the window
			for(int iDisparity = i;iDisparity<i+width;iDisparity++){
				for(int jDisparity = j;jDisparity<j+height;jDisparity++){
					disparity.at<uchar>(iDisparity,jDisparity) = dispar;
				}
			}
			// std::cout << "Finished Disparity filling\n";

			// break;
			// std::cout << j << '\n';
		}
		
		for(auto it = best_right_points.begin(); it<best_right_points.end();it++){
			Point p (it->x, it->y);
			// circle(D.I2, p, 2, Scalar(0,255,0), 2);
			// std::cout << p << '\n';
		}

		std::cout << i << '\n';
		// break;
	}
	// std::cout << "Finished dissimilarity calculation.\n";
	imshow("I1", D.I1);
	imshow("I2", D.I2);
	imshow("disparity", disparity);
	waitKey(0);

    cv::imwrite("../runs/disparity/disparity.jpg", disparity);
   	// TODO: disparity image or depth map?
}


int main(int argc, char** argv)
{
	Data D;
	D.I1 = imread("../data/box1.jpg");
	D.I2 = imread("../data/box2.jpg");

	// resizing
	Size size(D.I1.rows*0.25, D.I1.cols*0.25);
	resize(D.I1, D.I1, size);
	resize(D.I2, D.I2, size);

	// setting grayscale and floating point value versions of 2 stereo images
	cvtColor(D.I1, D.G1, COLOR_BGR2GRAY);
	cvtColor(D.I2, D.G2, COLOR_BGR2GRAY);
	D.G1.convertTo(D.F1, CV_32F);
	D.G2.convertTo(D.F2, CV_32F);

	// reading fundamental matrix from the file
	ifstream f("../runs/Fmatrix/Fmatrix.txt");
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			f >> D.F(i,j);
		}
	}
	std::cout << D.F << '\n';

	// calculating depth map
	calculate_depth_map(D);

	return 0;
}
