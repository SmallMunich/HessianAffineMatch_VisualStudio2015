// HessianAffineMatch.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include "hesaff_base.hpp"
#include <Windows.h>
#include "match_util.h"

using namespace cv;
using namespace std;
// 仿射特征提取
void features_extract(Mat& img, /*vector<KeyPoint> kpts, */char* outputfile);

void draw_kpts_ellipse(Mat& img, std::vector<KeyPoint>& kpts);


int main(void) 
{ 
  //-- 读入输入图像......
  Mat img1 = imread("..\\HessianAffineMatch\\image\\img1.png",1);
  Mat img2 = imread("..\\HessianAffineMatch\\image\\img2.png",1);
  //Mat img1 = imread("..\\HessianAffineMatch\\image\\adam_zoom1_front.png", 1);
  //Mat img2 = imread("..\\HessianAffineMatch\\image\\adam_zoom1_80deg.png", 1);
  //-- 特征点与描述子输出文件
  char* outputfile1 = "..\\HessianAffineMatch\\image\\desc11.txt";
  char* outputfile2 = "..\\HessianAffineMatch\\image\\desc22.txt";
  //-- 特征点提取函数......
  features_extract(img1, outputfile1);
  features_extract(img2, outputfile2);
  //draw_kpts_ellipse(img1, kpts1);
  //draw_kpts_ellipse(img2, kpts2);

  // 匹配模块参数输入输出......
  Mat img_com_dmatches = Mat(Size(img1.cols + img2.cols, MAX(img1.rows, img2.rows)), CV_8UC3);
  Mat img_com_inliers = Mat(Size(img1.cols + img2.cols, MAX(img1.rows, img2.rows)), CV_8UC3);
  
  vector<KeyPoint> kpts1, kpts2;
  Mat desc1, desc2, desc1_32, desc2_32;
  
  double t1 = 0.0, t2 = 0.0;
  t1 = getTickCount();
  //-- 打开文件获取img1和img2的特征点与描述子
  read_file_kpts_desc(outputfile1, kpts1, desc1);
  read_file_kpts_desc(outputfile2, kpts2, desc2);

  //-- 描述子归一化
  desc1.convertTo(desc1_32, CV_32F, 1.0 / 255.0, 0);
  desc2.convertTo(desc2_32, CV_32F, 1.0 / 255.0, 0);

  //-- BF_Match
  vector<vector<DMatch> > dmatches;
  vector<Point2f> matches, inliers;
  Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");
  //-- knnMatch 
  matcher_l2->knnMatch(desc1_32, desc2_32, dmatches, 2);
  //-- NNDR     DRATO: 0.8f
  matches2points_nndr(kpts1, kpts2, dmatches, matches, DRATIO);

  //-- RANSAC   MIN_H_ERROR: 2.50f
  Mat Homography;
  compute_inliers_ransac(matches, inliers, MIN_H_ERROR, Homography, false);

  //-- 内联点区分...
  vector<KeyPoint> kpts1_ellipse, kpts2_ellipse;
  divide_inliers_kpts(inliers, kpts1_ellipse, kpts2_ellipse);
  
  //-- 计算匹配RMSE误差
  double rmse = 0;
  compute_MatchImgRMSE(inliers, Homography, rmse);

  t2 = getTickCount();
  double matchTime = (t2 - t1) / getTickFrequency();

  //-- Additional message kpts
  compute_additional_message_kpts(kpts1, kpts1_ellipse);
  compute_additional_message_kpts(kpts2, kpts2_ellipse);

  //-- Draw the kpts' Ellipse
  draw_inliers_ellipse(img1, kpts1_ellipse);
  draw_inliers_ellipse(img2, kpts2_ellipse);

  //-- 计算内联点统计参数......
  int nmatches = matches.size() / 2;
  int ninliers = inliers.size() / 2;
  int noutliers = nmatches - ninliers;
  float ratio = 100.0*((float)ninliers / (float)nmatches);
  float ratio_eval = 100.0*((float)ninliers / ((float)min(kpts1.size(), kpts2.size())));

  cout << "Number of Matches: " << nmatches << endl;
  cout << "Number of Inliers: " << ninliers << endl;
  cout << "Number of Outliers: " << noutliers << endl;
  cout << "Inliers Ratio: " << ratio << endl;
  cout << "Match Desc Time is: " << matchTime << endl;
  cout << "Match evalute performance is " << rmse << endl << endl;
  //-- 参数输出文件......
  ofstream fout;
  fout.open("..\\HessianAffineMatch\\image\\result\\matches_result_11_22.txt", ios::out);
  fout << "Image Planar Transform Homography: " << endl;
  for (int i = 0; i<3; ++i)
  {
	  for (int j = 0; j<3; ++j)
	  {
		  fout << setw(25) << setiosflags(ios::right) << Homography.at<double>(i, j);
	  }
	  fout << endl;
  }
  fout << endl;

  fout << "Number of Matches: " << nmatches << endl;
  fout << "Number of Inliers: " << ninliers << endl;
  fout << "Number of Outliers: " << noutliers << endl;
  fout << "Inliers Ratio: " << ratio << endl;
  fout << "Match Desc Time is: " << matchTime << endl;
  fout << "Matches Ratio Eval: " << ratio_eval << endl;
  fout << "Eval performance: " << rmse << endl << endl;
  fout << flush;
  fout.close();


  //-- Draw the NNDR Match Result
  draw_inliers_difference(img1, img2, img_com_dmatches, matches);
  imshow("show_dmatches_match", img_com_dmatches);
  imwrite("..\\HessianAffineMatch\\image\\result\\inliers_dmatches_11_22.jpg", img_com_dmatches);

  //-- Draw the RANSAC Match Result
  draw_inliers_difference(img1, img2, img_com_inliers, inliers);
  imshow("show_inliers_match", img_com_inliers);
  imwrite("..\\HessianAffineMatch\\image\\result\\inliers_inliers_11_22.jpg", img_com_inliers);

  waitKey(0);

  system("pause");

  return 0;
}

void features_extract(Mat& img, /*vector<KeyPoint> kpts, */char* outputfile)
{
	assert(true != img.empty());
	Mat image(img.rows, img.cols, CV_32FC1, Scalar(0));

	float *out = image.ptr<float>(0);
	unsigned char *in = img.ptr<unsigned char>(0);

	if (img.channels() == 1)
	{
		for (size_t i = img.rows*img.cols; i > 0; i--)
		{
			*out = (float)in[0];
			out++;
			in++;
		}
	}
	else
	{
		for (size_t i = img.rows*img.cols; i > 0; i--)
		{
			*out = (float(in[0]) + in[1] + in[2]) / 3.0f;
			out++;
			in += 3;
		}
	}

	double startTime = getTickCount();

	HessianAffineParams par;
	double t1 = GetTickCount();
	{
		// copy params 
		PyramidParams p;
		p.threshold = par.threshold;

		AffineShapeParams ap;
		ap.maxIterations = par.max_iter;
		ap.patchSize = par.patch_size;
		ap.mrSize = par.desc_factor;

		SIFTDescriptorParams sp;
		sp.patchSize = par.patch_size;

		AffineHessianDetector detector(image, p, ap, sp);
		t1 = getTime(); g_numberOfPoints = 0;
		detector.detectPyramidKeypoints(image);
		cout << "Detected " << g_numberOfPoints << " keypoints and " << g_numberOfAffinePoints << " affine shapes in " << getTime() - t1 << " sec." << endl;

		// ostream out(std::cout.rdbuf());
		detector.exportKeypoints(outputfile);

		//Mat img_s = imread("..\\hessian_affine\\image\\speckle_sar.jpg",1);
		//vector<KeyPoint> kpts(detector.kpts);
		//draw_kpts_ellipse(img_s, kpts);
	}

	double endTime = getTickCount();
	double extractTime = (endTime - startTime) / getTickFrequency();
	cout << "FeaturesExtractor Time is " << extractTime << endl;
}

void draw_kpts_ellipse(Mat& img, std::vector<KeyPoint>& kpts)
{
	// void ellipse( Mat& img, Point center, Size axes, double angle, double startAngle, double endAngle
	//               const Scalar& color, int thickness=1, int lineType=8, int shift=0 );
	//-- img 图像 center 椭圆中心坐标  axes 轴的长度  angle 偏转角度  startAngle 圆弧起始角度
	//-- endAngle 圆弧终结角度   color 线条颜色  lineType 线条类型  shift 圆心坐标点和数轴的精度

	int size = kpts.size();
	double a=0, b=0, c=0;
	int major_semi_axis=0, minor_semi_axis=0;
	Size axes;

	for ( int i=0; i < size; ++i )
	{
		a = kpts[i].a;
		b = kpts[i].b;
		c = kpts[i].c;
		//-- 归一化仿射形状的长短轴  参考hessian_affine 中描述子生成计算

/*		double tr = a + c;
		double sqrtDet = (double)sqrt(a*c - b*b);
		double d = (a + sqrtDet) / sqrt(tr + 2.0*sqrtDet);
		double e = b / sqrt(tr + 2.0 * sqrtDet);
		double f = (c + sqrtDet) / sqrt(tr + 2.0 * sqrtDet);
*/
        // 二阶矩阵参数求取椭圆长短半轴和角度
	    double val1 = 0.5*((a+c)+ sqrt((a-c)*(a-c)+4*b*b));
	    double val2 = 0.5*((a+c)- sqrt((a-c)*(a-c)+4*b*b));

	    int semi_val1 = 1.0/sqrt(val1);
	    int semi_val2 = 1.0/sqrt(val2);

        if ( semi_val1 > semi_val2 )  {
			axes.width  = semi_val1;
			axes.height = semi_val2;
	  
	    }else  {
			axes.width  = semi_val2;
			axes.height = semi_val1;
	    }
		ellipse(img, Point(kpts[i].pt.x, kpts[i].pt.y), axes, kpts[i].angle*180/M_PI, 0, 360, CV_RGB(255,0,0), 1, 8, 0);
		circle(img, Point(kpts[i].pt.x, kpts[i].pt.y), 1, CV_RGB(0,255,255), 2);
	}

	imshow("draw_ellipse_image", img);
	imwrite("..\\HessianAffineMatch\\image\\result\\ellipse_image.jpg",img);
	// waitKey(0);
}





