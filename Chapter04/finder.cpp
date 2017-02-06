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

#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "contentFinder.h"
#include "colorhistogram.h"

int main()
{
	// Read reference image
	cv::Mat image= cv::imread("baboon01.jpg");
	if (!image.data)
		return 0; 

	// initial window position
	cv::Rect rect(110, 45, 35, 45);
	cv::rectangle(image, rect, cv::Scalar(0, 0, 255));

	// Baboon's face ROI
	cv::Mat imageROI = image(rect);

	cv::namedWindow("Image 1");
	cv::imshow("Image 1",image);

	// Get the Hue histogram of the Baboon's face
	int minSat=65;
	ColorHistogram hc;
	cv::Mat colorhist= hc.getHueHistogram(imageROI,minSat);

	ContentFinder finder;
	finder.setHistogram(colorhist);
	finder.setThreshold(0.2f);

	// Convert to HSV space (just for display)
	cv::Mat hsv;
	cv::cvtColor(image, hsv, CV_BGR2HSV);

	// Split the image
	vector<cv::Mat> v;
	cv::split(hsv,v);

	// Eliminate pixels with low saturation
	cv::threshold(v[1],v[1],minSat,255,cv::THRESH_BINARY);
	cv::namedWindow("Saturation mask");
	cv::imshow("Saturation mask",v[1]);

	//--------------
	// Second image
	image= cv::imread("baboon02.jpg");

	cv::namedWindow("Image 2");
	cv::imshow("Image 2",image);

	// Convert to HSV space
	cv::cvtColor(image, hsv, CV_BGR2HSV);

	// Get back-projection of hue histogram
	int ch[1]={0};
	finder.setThreshold(-1.0f); // no thresholding
	cv::Mat result= finder.find(hsv,0.0f,180.0f,ch);

	// Display back projection result
	cv::namedWindow("Backprojection on second image");
	cv::imshow("Backprojection on second image",result);

	// initial window position
	cv::rectangle(image, rect, cv::Scalar(0,0,255));

	// search objet with mean shift
	cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
		10,     // iterate max 10 times
		1);     // or until the change in centroid position is less than 1px
	cout << "meanshift= " << cv::meanShift(result,rect,criteria) << endl;

	// draw output window
	cv::rectangle(image, rect, cv::Scalar(0,255,0));

	// Display image
	cv::namedWindow("Image 2 result");
	cv::imshow("Image 2 result",image);

	cv::waitKey();
	return 0;
}	