/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "locker.hpp"
#include "..\\HessianAffineMatch\\hessaff\\pyramid.h"
#include "..\\HessianAffineMatch\\hessaff\\helpers.h"
#include "..\\HessianAffineMatch\\hessaff\\affine.h"
#include "..\\HessianAffineMatch\\hessaff\\siftdesc.h"

using namespace cv;
using namespace std;

struct HessianAffineParams
{
  float threshold;
  int   max_iter;
  float desc_factor;
  int   patch_size;
  bool  verbose;
  HessianAffineParams()
  {
    threshold = 16.0f/3.0f;
    max_iter = 16;
    desc_factor = 3.0f*sqrt(3.0f);
    patch_size = 41;
    verbose = false;
  }
};

int g_numberOfPoints = 0;
int g_numberOfAffinePoints = 0;

struct Keypoint
{
  float x, y, s;
  float a11,a12,a21,a22;
  float response;
  int type;
  unsigned char desc[128];
};

struct AffineHessianDetector : public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
  const Mat image;
  SIFTDescriptor sift;
  vector<Keypoint> keys;
  vector<KeyPoint> kpts;

public:
  AffineHessianDetector(const Mat &image, const PyramidParams &par, const AffineShapeParams &ap, const SIFTDescriptorParams &sp) : 
    HessianDetector(par), 
    AffineShape(ap), 
    image(image),
    sift(sp)
  {
    this->setHessianKeypointCallback(this);
    this->setAffineShapeCallback(this);
  }

  void onHessianKeypointDetected(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
  {
    g_numberOfPoints++;
    findAffineShape(blur, x, y, s, pixelDistance, type, response);
  }

  void onAffineShapeFound(
      const Mat &blur, float x, float y, float s, float pixelDistance,
      float a11, float a12,
      float a21, float a22, 
      int type, float response, int iters) 
  {
    // convert shape into a up is up frame
    rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);

    // now sample the patch
    if (!normalizeAffine(image, x, y, s, a11, a12, a21, a22))
    {
      // compute SIFT
      sift.computeSiftDescriptor(this->patch);
      // store the keypoint
      keys.push_back(Keypoint());
      Keypoint &k = keys.back();
      k.x = x; k.y = y; k.s = s; k.a11 = a11; k.a12 = a12; k.a21 = a21; k.a22 = a22; k.response = response; k.type = type;
      for (int i=0; i<128; i++)
        k.desc[i] = (unsigned char)sift.vec[i];
      // debugging stuff
      if (0)
      {
        cout << "x: " << x << ", y: " << y
          << ", s: " << s << ", pd: " << pixelDistance
          << ", a11: " << a11 << ", a12: " << a12 << ", a21: " << a21 << ", a22: " << a22 
          << ", t: " << type << ", r: " << response << endl; 
        for (size_t i=0; i<sift.vec.size(); i++)
          cout << " " << sift.vec[i];
        cout << endl;
      }
      g_numberOfAffinePoints++;
    }
  }
//void exportKeypoints(ostream &out)
  void exportKeypoints(char* data_file)
  {
    //out << 128 << endl;
    //out << keys.size() << endl;

    //Mat draw_ellipse = imread("..\\hessian_affine\\image\\graf\\img1.jpg",1);

	ofstream out(data_file);

	KeyPoint kpt;

    for (size_t i=0; i<keys.size(); i++)
    {
      Keypoint &k = keys[i];

      float sc = AffineShape::par.mrSize * k.s;
      Mat A = (Mat_<float>(2,2) << k.a11, k.a12, k.a21, k.a22);



      SVD svd(A, SVD::FULL_UV);

      float *d = (float *)svd.w.data;
      d[0] = 1.0f/(d[0]*d[0]*sc*sc);
      d[1] = 1.0f/(d[1]*d[1]*sc*sc);

      A = svd.u * Mat::diag(svd.w) * svd.u.t();
	  // kpts coordinate 
	  // matrix:     [a b; b c]
	  double a =  A.at<float>(0,0);
	  double b =  A.at<float>(0,1);
	  double c =  A.at<float>(1,1);

	  // 区域椭圆长半轴与水平方向的角度
	  double angle = atan(b/(a-c))/2;

      out << k.x << " " << k.y << " " << A.at<float>(0,0) << " " << A.at<float>(0,1) << " " << A.at<float>(1,1) << " " << angle;

	  // peng-wu  2017/03/28
	  kpt.pt.x  = k.x;
	  kpt.pt.y  = k.y;
	  kpt.angle = angle;
	  kpt.a     = A.at<float>(0,0);
	  kpt.b     = A.at<float>(0,1);
	  kpt.c     = A.at<float>(1,1);

	  kpts.push_back(kpt);

	  // descriptors 
      for (size_t i=0; i<128; i++)
        out << " " << int(k.desc[i]);
      out << endl;


	  // 二阶矩阵参数求取椭圆长短半轴和角度
	  double val1 = 0.5*((a+c)+ sqrt((a-c)*(a-c)+4*b*b));
	  double val2 = 0.5*((a+c)- sqrt((a-c)*(a-c)+4*b*b));

	  int semi_val1 = 1.0/sqrt(val1);
	  int semi_val2 = 1.0/sqrt(val2);

	  Point center;
	  center.x = k.x;
	  center.y = k.y;

	  Size axes;
	  if ( semi_val1 > semi_val2 )  {
	     axes.width  = semi_val1;
	     axes.height = semi_val2;
	  
	  }else  {
		 axes.width  = semi_val2;
	     axes.height = semi_val1;
	  }

	  //ellipse(draw_ellipse, center, axes, angle*180/M_PI, 0, 360, CV_RGB(255,255,255), 2, 8, 0);
	  //circle(draw_ellipse,  center, 1, CV_RGB(255,255,255), 2);

    }

	//imshow("draw_ellipse", draw_ellipse);
	//waitKey(0);

  }
};

