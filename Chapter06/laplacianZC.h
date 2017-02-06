/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 6 of the cookbook:  
   Computer Vision Programming using the OpenCV Library 
   Second Edition 
   by Robert Laganiere, Packt Publishing, 2013.

   This program is free software; permission is hereby granted to use, copy, modify, 
   and distribute this source code, or portions thereof, for any purpose, without fee, 
   subject to the restriction that the copyright notice may not be removed 
   or altered from any source or altered source distribution. 
   The software is released on an as-is basis and without any warranties of any kind. 
   In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
   The author disclaims all warranties with regard to this software, any use, 
   and any consequent failure, is purely the responsibility of the user.
 
   Copyright (C) 2013 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#if !defined LAPLACEZC
#define LAPLACEZC

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class LaplacianZC {

  private:
	  // laplacian
	  cv::Mat laplace;

	  // Aperture size of the laplacian kernel
	  int aperture;

  public:

	  LaplacianZC() : aperture(3) {}

	  // Set the aperture size of the kernel
	  void setAperture(int a) {

		  aperture= a;
	  }

	  // Get the aperture size of the kernel
	  int getAperture() const {

		  return aperture;
	  }

	  // Compute the floating point Laplacian
	  cv::Mat computeLaplacian(const cv::Mat& image) {

		  // Compute Laplacian
		  cv::Laplacian(image,laplace,CV_32F,aperture);

		  return laplace;
	  }

	  // Get the Laplacian result in 8-bit image 
	  // zero corresponds to gray level 128
	  // if no scale is provided, then the max value will be
	  // scaled to intensity 255
	  // You must call computeLaplacian before calling this method
	  cv::Mat getLaplacianImage(double scale=-1.0) {

		  if (scale<0) {
	
			  double lapmin, lapmax;
			  cv::minMaxLoc(laplace,&lapmin,&lapmax);

			  scale= 127/ std::max(-lapmin,lapmax);
		  }

		  cv::Mat laplaceImage;
		  laplace.convertTo(laplaceImage,CV_8U,scale,128);

		  return laplaceImage;
	  }

	  // Get a binary image of the zero-crossings
	  // laplacian image should be CV_32F
	  cv::Mat getZeroCrossings(cv::Mat laplace) {

		  // threshold at 0
		  // negative values in black
		  // positive values in white
		  cv::Mat signImage;
		  cv::threshold(laplace,signImage,0,255,cv::THRESH_BINARY);

		  // convert the +/- image into CV_8U
		  cv::Mat binary;
		  signImage.convertTo(binary,CV_8U);

		  // dilate the binary image of +/- regions
		  cv::Mat dilated;
		  cv::dilate(binary,dilated,cv::Mat());
	
		  // return the zero-crossing contours
		  return dilated-binary;
	  }
};


#endif
