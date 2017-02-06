/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 4 of the book:
OpenCV3 Computer Vision Application Programming Cookbook
Third Edition
by Robert Laganiere, Packt Publishing, 2016.

This program is free software; permission is hereby granted to use, copy, modify,
and distribute this source code, or portions thereof, for any purpose, without fee,
subject to the restriction that the copyright notice may not be removed
or altered from any source or altered source distribution.
The software is released on an as-is basis and without any warranties of any kind.
In particular, the software is not guaranteed to be fault-tolerant or free from failure.
The author disclaims all warranties with regard to this software, any use,
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2016 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <math.h>

// remapping an image by creating wave effects
void wave(const cv::Mat &image, cv::Mat &result) {

	// the map functions
	cv::Mat srcX(image.rows,image.cols,CV_32F); // x-map
	cv::Mat srcY(image.rows,image.cols,CV_32F); // y-map

	// creating the mapping
	for (int i=0; i<image.rows; i++) {
		for (int j=0; j<image.cols; j++) {

			srcX.at<float>(i,j)= j;
			srcY.at<float>(i,j)= i+3*sin(j/6.0);

			// horizontal flipping
			// srcX.at<float>(i,j)= image.cols-j-1;
			// srcY.at<float>(i,j)= i;
		}
	}

	// applying the mapping
	cv::remap(image,  // source image
		      result, // destination image
			  srcX,   // x map
			  srcY,   // y map
			  cv::INTER_LINEAR); // interpolation method
}

int main()
{
	// open image
	cv::Mat image= cv::imread("boldt.jpg",0);

	cv::namedWindow("Image");
	cv::imshow("Image",image);

	// remap image
	cv::Mat result;
	wave(image,result);

	cv::namedWindow("Remapped image");
	cv::imshow("Remapped image",result);

	cv::waitKey();
	return 0;
}


