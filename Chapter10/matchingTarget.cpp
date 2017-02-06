/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 10 of the book:
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

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "targetMatcher.h"

int main()
{
	// Read input images
	cv::Mat target= cv::imread("cookbook1.jpg",0);
	cv::Mat image= cv::imread("objects.jpg",0);
	if (!target.data || !image.data)
		return 0; 

    // Display the images
	cv::namedWindow("Target");
	cv::imshow("Target",target);
	cv::namedWindow("Image");
	cv::imshow("Image", image);

	// Prepare the matcher 
	TargetMatcher tmatcher(cv::FastFeatureDetector::create(10),cv::BRISK::create());
	tmatcher.setNormType(cv::NORM_HAMMING);

	// definition of the output data
	std::vector<cv::DMatch> matches;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector<cv::Point2f> corners;

	// set the target image
	tmatcher.setTarget(target); 

	// match image with target
	tmatcher.detectTarget(image, corners);
	// draw the target corners on the image
	if (corners.size() == 4) { // we have a detection

		cv::line(image, cv::Point(corners[0]), cv::Point(corners[1]), cv::Scalar(255, 255, 255), 3);
		cv::line(image, cv::Point(corners[1]), cv::Point(corners[2]), cv::Scalar(255, 255, 255), 3);
		cv::line(image, cv::Point(corners[2]), cv::Point(corners[3]), cv::Scalar(255, 255, 255), 3);
		cv::line(image, cv::Point(corners[3]), cv::Point(corners[0]), cv::Scalar(255, 255, 255), 3);
	}
	cv::namedWindow("Target detection");
	cv::imshow("Target detection",image);

	cv::waitKey();
	return 0;
}