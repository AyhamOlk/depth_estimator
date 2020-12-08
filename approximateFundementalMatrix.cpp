#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

//THIS code uses the 8 point algorithm to approximate the fundamental matrix

Mat eight_point(Mat left, Mat right, Mat &F, int height1, int width1, int height2, int width2){
    /*
    1- Build the constraint matrix A from observations
    2- Compute [U,D,V] = svd(A)
    3- Extract fundamental matrix from the column of V corresponding to the smallest
    singular value.
    4- reshape this 9D vector to a 3x3 matrix
    5- Enforce rank2 constraint using svd by zeroing the smallest eigenvalue
    [U,D,V] = svd(F)
    F = U* diag(D(0), D(1), 0)*V^T;
    */
    // Construction of A
    // std::cout << right << std::endl;

	std::cout<<left<<std::endl;
	std::cout<<right<<std::endl;
	return F;

	//normalization
	Mat T = Mat::zeros(3,3,CV_32F); Mat T_prime = Mat::zeros(3,3,CV_32F);
    T.at<float>(0,0) = 2.0/width1;
    T.at<float>(0,2) = -1;
    T.at<float>(1,1) = 2.0/height1;
    T.at<float>(1,2) = -1;
    T.at<float>(2,2) = 1;
    T_prime.at<float>(0,0) = 2.0/width2;
    T_prime.at<float>(0,2) = -1;
    T_prime.at<float>(1,1) = 2.0/height2;
    T_prime.at<float>(1,2) = -1;
    T_prime.at<float>(2,2) = 1;

    Mat A = Mat::ones(left.rows, 9, CV_32F);
    for(int i=0;i<A.rows;i++){

		left.at<int>(i,0) = 2*left.at<int>(i,0)/height1 - 1;
        left.at<int>(i,1) = 2*left.at<int>(i,1)/width1 - 1;
        right.at<int>(i,0) = 2*right.at<int>(i,0)/height2 - 1;
        right.at<int>(i,1) = 2*right.at<int>(i,1)/width2 - 1;

        A.at<float>(i,8) = 1; // last element is 1
        A.at<float>(i,7) = left.at<int>(i,1);
        A.at<float>(i,6) = left.at<int>(i,0);
        A.at<float>(i,5) = right.at<int>(i,1); // the one in the right image
        A.at<float>(i,4) = left.at<int>(i,1)*right.at<int>(i,1);
        A.at<float>(i,3) = left.at<int>(i,0)*right.at<int>(i,1);
        A.at<float>(i,2) = right.at<int>(i,0);
        A.at<float>(i,1) = left.at<int>(i,1)*right.at<int>(i,0);
        A.at<float>(i,0) = left.at<int>(i,0)*right.at<int>(i,0);
    }

    // 3- Extract fundamental matrix from the column of V corresponding to the smallest
    //std::cout << A << std::endl;
    // Compute SVD
    Mat S, U, VT; // S contains the eigen values
    Mat A_AT = A.t()*A;
    SVDecomp(A_AT, S, U, VT, cv::SVD::FULL_UV);        //TODO A*AT or AT*A
    // std::cout << S << std::endl;
    // std::cout << VT << std::endl;
    // std::cout << S.at<float>(S.rows-1, 0) << std::endl;
    int index = S.rows - 1;
    // std::cout << VT.row(index) << std::endl;

    // 3- Extract fundamental matrix from the column of V corresponding to the smallest singular value.
    Mat smallest_eigen_vec = VT.row(index);

    //std::cout<<"V is :"<<VT.t()<<std::endl;
    //std::cout<<"smalles eigne vec is :"<<smallest_eigen_vec;
    // 4- reshape this 9D vector to a 3x3 matrix
    //Mat F = Mat::zeros(3,3,CV_32F);
    int counter = 0;
    for(int i=0;i<9;i++){
        F.at<float>(counter, i%3) = smallest_eigen_vec.at<float>(0,i);
        if ( (i + 1)%3 == 0 )
            counter++;
    }
    std::cout << F << std::endl;

    // 5- Enforce rank2 constraint using svd by zeroing the smallest eigenvalue
    Mat S1, U1, VT1; // S contains the eigen values
    SVDecomp(F, S1, U1, VT1, cv::SVD::FULL_UV);
    Mat diag = Mat::zeros(3,3,CV_32F);
    diag.at<float> (0,0) = S1.at<float>(0,0);
    diag.at<float> (1,1) = S1.at<float>(1,0);
    F = U1*diag*VT1;

    std::cout << F << std::endl;


    //std::cout << "S1 = " << S1 << std::endl;
    //std::cout << "diag = " << diag << std::endl;

	//std::cout<<"\n\n The F matrix you are looking for is: \n\n";
    





    F = T_prime.t()*F*T;
    std::cout<<F<<std::endl;
	return F;
}


int main()
{
	string expName = "faces_5";

	//Mat I_left = imread("../runs/detect/exp/cropped1.jpg", IMREAD_GRAYSCALE);
	//Mat I_right = imread("../runs/detect/exp/cropped2.jpg", IMREAD_GRAYSCALE);
//	Mat I_left = imread("../data/car1.jpeg", IMREAD_GRAYSCALE);
//	Mat I_right = imread("../data/car1.jpeg", IMREAD_GRAYSCALE);


	Mat I_left = imread("../data/face00.tif", IMREAD_GRAYSCALE);

	int height1 = I_left.rows; 
	int width1 = I_left.cols;



	Mat I_right = imread("../data/face01.tif", IMREAD_GRAYSCALE);

	int height2 = I_right.rows;
	int width2 = I_right.cols;

	std::cout<<height1<<" "<<width1<< " "<<height1<<" "<<height2<<"\n";
	//imshow("I_left", I_left);
	//imshow("I_right", I_right);

	vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
	
	Ptr<AKAZE> akaze = AKAZE::create();

	akaze->detectAndCompute(I_left, noArray(), kpts1, desc1);
    akaze->detectAndCompute(I_right, noArray(), kpts2, desc2);

	
	Mat J, J2;
	drawKeypoints( I_left, kpts1, J);
	//imshow("Feature I_left", J);
	drawKeypoints( I_right, kpts2, J2);
	//imshow("Feature I_right", J2);

	
	//compute feature correspondences
	BFMatcher M(NORM_HAMMING); 	

	// drawMatches between images
	vector< vector<DMatch> > nn_matches;
	vector< DMatch > matches;
	M.match(desc1, desc2, matches);
	

	Mat res;
    drawMatches(I_left, kpts1, I_right, kpts2, matches, res);
    //imshow("Knn_Matcher", res);
	

	std::vector<Point2f> matches1, matches2;
	//Fill matches as pt1 to pt2 
	for(  int i = 0; i < matches.size(); i++ ) //size_t
    {
		int idx1 = matches[i].queryIdx;
    	int idx2 = matches[i].trainIdx;
		
    	matches1.push_back(kpts1[idx1].pt);
    	matches2.push_back(kpts2[idx2].pt);
    }

	//mask points we need to remove as not perfect match we got
	Mat mask;

    Mat H = findHomography( matches1, matches2, RANSAC, 3.0, mask );

	int counter = 0;
	int counter_good_matches = 0;

	//Only take stuff we need, and here counting to know size of correct matches
	for(int i =0; i<matches.size(); i++) {
		if(mask.at<bool>(i,0) == true){ 
			counter_good_matches++;
		}
	}


	//depends on size of matches we have thus use counter_good_matches
	Mat left = Mat::zeros(counter_good_matches, 2, CV_32SC1); 
	Mat right = Mat::zeros(counter_good_matches, 2, CV_32SC1);

	vector< DMatch > img_matches; 

	for(int i =0; i<matches.size(); i++) {

		if(mask.at<bool>(i,0) == true){
			left.at<int>(counter,0) = matches1[i].x;
			left.at<int>(counter,1) = matches1[i].y;

			right.at<int>(counter,0) = matches2[i].x;
			right.at<int>(counter,1) = matches2[i].y;

			//std::cout<<counter<<" Match between: "<<matches1[i]<<"  ";
			//std::cout<<matches2[i]<<"\n";
			counter++;
			img_matches.push_back(matches[i]); //push as pair of matches x,y x2,y2
		}
		
	}

	Mat resGoodMatches;
    drawMatches(I_left, kpts1, I_right, kpts2, img_matches, resGoodMatches);
    //imshow("RANSAC", resGoodMatches);
	



	//Fundemental Matrix is 3by3s
	Mat F = Mat::zeros(3,3,CV_32F);
	F = eight_point(left, right, F, height1, width1, height2, width2);

	string pathName = "../runs/descriptors_F_Matrix/"+expName;
	//std::cout<<F;
	
	if (mkdir(pathName.c_str(), 0777) == -1) {
		cerr << "Error :  " << strerror(errno) << endl; 
	}
	else {
		cout << "Directory created\n"; 
		imwrite(pathName+"/Features_left.jpg", J);
		
		imwrite(pathName+"/Features_right.jpg", J2);
		imwrite(pathName+"/KNN_matches.jpg", res);
		imwrite(pathName+"/RANSAC_good_matches.jpg", resGoodMatches);
		ofstream myfile;
 		myfile.open (pathName+"/F_Matrix.txt");
  		myfile << "Fundamental Matrix Of Experiment: ";
		myfile << expName;
		myfile << " --> is:\n\n";
		myfile << F;
  		myfile.close();
		
	}
		
	//waitKey(0);
	return 0;
}


