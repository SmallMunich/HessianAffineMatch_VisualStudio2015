/************************************************************************************
*    FileName                :   match_util.h/match_util.cpp
*    Copyright               :   XX University, All Rights Reserved.
*
*    Create Date             :   2017/01/10
*    Author                  :   Zeya Wu
*	 Abstraction Description :   this head file is used for mser_sift to match 
*								 planar images
*    Version                 :   No.1
*************************************************************************************/


#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const float MIN_H_ERROR = 2.50f;	// Maximum error in pixels to accept an inlier
const float DRATIO      = 0.80f;    // NNDR Matching value
const double M_PI         = 3.141592654;

void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
                         const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch> >& matches,
                         std::vector<cv::Point2f>& pmatches, float nndr) ;

void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
                            std::vector<cv::Point2f>& inliers,
                            float error,  cv::Mat& Homo, bool use_fund) ;

void draw_inliers_difference(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
							 const std::vector<cv::Point2f>& ptpairs );

void draw_inliers_ellipse(Mat& img, std::vector<KeyPoint>& kpts);

//-- compute the file rows
int compute_file_count_lines(const char* filename);

//-- read the kpts and descriptors data from file
int read_file_kpts_desc(const char* filename, std::vector<KeyPoint>& kpts, Mat& desc);


//--  features2d.hpp 
//--  class CV_EXPORTS_W_SIMPLE KeyPoint
//--  	CV_PROP_RW double a;
//--	CV_PROP_RW double b;
//--	CV_PROP_RW double c;
void divide_inliers_kpts(std::vector<Point2f> inliers, std::vector<KeyPoint>& kpts1_inliers, std::vector<KeyPoint>& kpts2_inliers);

void compute_additional_message_kpts(std::vector<KeyPoint>& kpts, std::vector<KeyPoint>& kpts_inliers);

void compute_MatchImgRMSE(std::vector<Point2f>& inliers, Mat& Homography, double& rmse);