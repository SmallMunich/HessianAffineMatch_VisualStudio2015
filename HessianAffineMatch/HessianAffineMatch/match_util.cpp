
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
#include "stdafx.h"
#include "match_util.h"


void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
                         const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch> >& matches,
                         std::vector<cv::Point2f>& pmatches, float nndr) 
{

  float dist1 = 0.0, dist2 = 0.0;

  for (size_t i = 0; i < matches.size(); i++) 
  {
		DMatch dmatch = matches[i][0];
		dist1 = matches[i][0].distance;
		dist2 = matches[i][1].distance;

		if (dist1 < nndr*dist2)
		{
			  pmatches.push_back(train[dmatch.queryIdx].pt);
			  pmatches.push_back(query[dmatch.trainIdx].pt);
		}
  }
}


void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
                            std::vector<cv::Point2f>& inliers,
                            float error,  cv::Mat& Homo, bool use_fund) 
{

  vector<Point2f> points1, points2;
  Mat H = Mat::zeros(3,3,CV_32F);
  int npoints = matches.size()/2;
  Mat status = Mat::zeros(npoints,1,CV_8UC1);

  for (size_t i = 0; i < matches.size(); i+=2) 
  {
		points1.push_back(matches[i]);
		points2.push_back(matches[i+1]);
  }
  // add the judge condition to prevent the input paramers wrong in findFundamentalMat() or findHomography() calculate
  if (4 > points1.size() || 4 >  points2.size())
	  return ;

  if (use_fund == true)
  {
		H = findFundamentalMat(points1,points2,CV_FM_RANSAC,error,0.99,status);
  }
  else 
  {
		H = findHomography(points1,points2,CV_RANSAC,error,status);
  }
  Homo = H;
  for (int i = 0; i < npoints; i++) 
  {
		if (status.at<unsigned char>(i) == 1) 
		{
			  inliers.push_back(points1[i]);
			  inliers.push_back(points2[i]);
		}
  }

  //for(int i=0; i<3; ++i)
  //{
		//for(int j=0; j<3; ++j)
		//{
		//	cout << "  " << H.at<double>(i,j);
		//}
		//cout << endl;
  //}


}


void draw_inliers_difference(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
							 const std::vector<cv::Point2f>& ptpairs)
{
	  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	  float rows1 = 0.0, cols1 = 0.0;
	  float rows2 = 0.0, cols2 = 0.0;
	  float ufactor = 0.0, vfactor = 0.0;

	  rows1 = img1.rows;	cols1 = img1.cols;
	  rows2 = img2.rows;	cols2 = img2.cols;

	  cv::Mat img_aux = cv::Mat(cv::Size(img1.cols,img1.rows), CV_8UC3);

	  if ( rows1 >= rows2 )	{
		  for (int i = 0; i < img_com.rows; ++i) 
		  {
			for (int j=0; j < img_com.cols; ++j)
			{
				 if (j < img1.cols) 
				 {
					*(img_com.ptr<unsigned char>(i)+3*j)   = *(img1.ptr<unsigned char>(i)+3*j);
					*(img_com.ptr<unsigned char>(i)+3*j+1) = *(img1.ptr<unsigned char>(i)+3*j+1);
					*(img_com.ptr<unsigned char>(i)+3*j+2) = *(img1.ptr<unsigned char>(i)+3*j+2);
				  }
				 else if( j >= img1.cols && i < rows2)	
				 {
					*(img_com.ptr<unsigned char>(i)+3*j)   = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols));
					*(img_com.ptr<unsigned char>(i)+3*j+1) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols)+1);
					*(img_com.ptr<unsigned char>(i)+3*j+2) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols)+2); 
				 }
				 else 
				 {
					*(img_com.ptr<unsigned char>(i)+3*j)   = 0;
					*(img_com.ptr<unsigned char>(i)+3*j+1) = 0;
					*(img_com.ptr<unsigned char>(i)+3*j+2) = 0;
				 }
			}  
		  }

		  for (size_t i = 0; i < ptpairs.size(); i+= 2) 
		  {
			 x1 = (int)(ptpairs[i].x+.5);
			 y1 = (int)(ptpairs[i].y+.5);
			 x2 = (int)(ptpairs[i+1].x+img1.cols+.5);
			 y2 = (int)(ptpairs[i+1].y+.5);
			 cv::line(img_com, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(255,255,255),1);
		  }
	  }
	  else	{
		  for (int i = 0; i < img_com.rows; ++i) 
		  {
			for (int j=0; j < img_com.cols; ++j)
			{
				if (j < img1.cols && i < img1.rows) 
				 {
					*(img_com.ptr<unsigned char>(i)+3*j)   = *(img1.ptr<unsigned char>(i)+3*j);
					*(img_com.ptr<unsigned char>(i)+3*j+1) = *(img1.ptr<unsigned char>(i)+3*j+1);
					*(img_com.ptr<unsigned char>(i)+3*j+2) = *(img1.ptr<unsigned char>(i)+3*j+2);
				  }
				 else if( j >= img1.cols )	
				 {
					*(img_com.ptr<unsigned char>(i)+3*j)   = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols));
					*(img_com.ptr<unsigned char>(i)+3*j+1) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols)+1);
					*(img_com.ptr<unsigned char>(i)+3*j+2) = *(img2.ptr<unsigned char>(i)+3*(j-img_aux.cols)+2); 
				 }
				 else 
				 {
					*(img_com.ptr<unsigned char>(i)+3*j)   = 0;
					*(img_com.ptr<unsigned char>(i)+3*j+1) = 0;
					*(img_com.ptr<unsigned char>(i)+3*j+2) = 0;
				 }
			}  
		  }

		  for (size_t i = 0; i < ptpairs.size(); i+= 2) 
		  {
			 x1 = (int)(ptpairs[i].x+.5);
			 y1 = (int)(ptpairs[i].y+.5);
			 x2 = (int)(ptpairs[i+1].x+img1.cols+.5);
			 y2 = (int)(ptpairs[i+1].y+.5);
			 cv::line(img_com, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(255,255,255),1);
		  }

	  }
}

//-- Affine Normalized Ellipse Shape 
void draw_inliers_ellipse(Mat& img, std::vector<KeyPoint>& kpts)
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
		ellipse(img, Point(kpts[i].pt.x, kpts[i].pt.y), axes, kpts[i].angle*180/M_PI, 0, 360, CV_RGB(255,255,255), 2, 8, 0);
		circle(img, Point(kpts[i].pt.x, kpts[i].pt.y), 1, CV_RGB(255,255,255), 2);
	}

}


int compute_file_count_lines(const char* filename)
{
	ifstream read_file;
	int num = 0;
	string temp;
	read_file.open(filename);

	if ( read_file.fail() )
	{
		cout << "File open false!" << endl;
		return -1;
	}
	else
	{
		while( getline(read_file, temp) )
		{
			++num;
		}
		return num;
	}
	read_file.close();
}


int read_file_kpts_desc(const char* filename, std::vector<KeyPoint>& kpts, Mat& desc)
{
	ifstream read_file;
	read_file.open(filename, ios::in);
	if ( read_file.fail() )
	{
		cout << "desc file text read failed!" << endl;
		return 0;
	}

	int num = compute_file_count_lines(filename);

	//--  特征点对应128维SIFT描述子
	desc = Mat::zeros(num, 128, CV_32FC1);  // initialize matrix

	double kpt_[6] = { 0 };
	KeyPoint kpt;
	for(int i=0; i<num; ++i)
	{
		for(int j=0; j<134; ++j)
		{
			if( j<6 )
			{
				read_file >> kpt_[j];
			}
			else
			{
				read_file >> desc.at<float>(i,j-6);
			}
			kpt.pt.x  = kpt_[0];   // kpts location : x
			kpt.pt.y  = kpt_[1];   // kpts location : y
			kpt.angle = kpt_[5];   // kpts location : angle

			kpt.a     = kpt_[2];   // Covariance matrix [a b; b c]
			kpt.b     = kpt_[3];
			kpt.c     = kpt_[4];
		}
		kpts.push_back(kpt);
	}

	return 1;
}


void divide_inliers_kpts(std::vector<Point2f> inliers, std::vector<KeyPoint>& kpts1_inliers, std::vector<KeyPoint>& kpts2_inliers)
{
	vector<Point2f>::iterator it;

	KeyPoint kpts;

	for(it=inliers.begin(); it != inliers.end(); ++it)
	{
		kpts.pt.x = (*it).x;
		kpts.pt.y = (*it).y;
		kpts1_inliers.push_back(kpts);

		it = it + 1;
		kpts.pt.x = (*it).x;
		kpts.pt.y = (*it).y;

		kpts2_inliers.push_back(kpts);
	}
}


void compute_additional_message_kpts(std::vector<KeyPoint>& kpts, std::vector<KeyPoint>& kpts_inliers)
{
	int inliers_size = kpts_inliers.size();
	int size = kpts.size();

	for(int i=0; i<inliers_size; ++i)
	{
		for(int j=0; j<size; ++j)
		{
			if( (kpts_inliers[i].pt.x == kpts[j].pt.x) && (kpts_inliers[i].pt.y == kpts[j].pt.y) )
			{
				kpts_inliers[i].a     = kpts[j].a;
				kpts_inliers[i].b     = kpts[j].b;
				kpts_inliers[i].c     = kpts[j].c;
				kpts_inliers[i].angle = kpts[j].angle;
				break;
			}
		}
	}
}


void compute_MatchImgRMSE(std::vector<Point2f>& inliers, Mat& Homography, double& rmse)
{
	vector<KeyPoint> kpts1, kpts2;
	//分开内联点对数据结构
	divide_inliers_kpts(inliers, kpts1, kpts2);

	int num = kpts1.size();

	vector<KeyPoint> xkpts1;

	Mat InImg = Mat(1, 3, CV_64FC1);
	Mat HomoT, OutImg;

	HomoT = Homography.t();

	KeyPoint data;
	vector<KeyPoint>::iterator  it;
	it = kpts1.begin();

	for ( it = kpts1.begin(); it != kpts1.end(); it++ )
	{
		InImg.at<double>(0,0) = (*it).pt.x;
		InImg.at<double>(0,1) = (*it).pt.y;
		InImg.at<double>(0,2) = 1;

		OutImg = InImg * HomoT;
	
	    data.pt.x = OutImg.at<double>(0,0)/OutImg.at<double>(0,2);
	    data.pt.y = OutImg.at<double>(0,1)/OutImg.at<double>(0,2);

	    xkpts1.push_back(data);
	}
	
	vector<KeyPoint>::iterator it1,it2;
	it1 = xkpts1.begin();
	it2 = kpts2.begin();
	double RMSE = 0;
	for ( it1 = xkpts1.begin(); it1 != xkpts1.end(); it1++ ) 
	{
		RMSE += ((*it1).pt.x-(*it2).pt.x)*((*it1).pt.x-(*it2).pt.x) + ((*it1).pt.y-(*it2).pt.y)*((*it1).pt.y-(*it2).pt.y);
		it2++;
	}

	rmse = sqrt(RMSE/num);

}