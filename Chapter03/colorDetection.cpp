/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 3 of the book:
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

#include "colordetector.h"

int main()
{
    // 1. Create image processor object
	ColorDetector cdetect;

    // 2. Read input image
	cv::Mat image= cv::imread("boldt.jpg");
	if (image.empty())
		return 0; 
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", image);

    // 3. Set input parameters
	cdetect.setTargetColor(230,190,130); // here blue sky

    // 4. Process the image and display the result
	cv::namedWindow("result");
	cv::Mat result = cdetect.process(image);
	cv::imshow("result",result);

	// or using functor
	// here distance is measured with the Lab color space
	ColorDetector colordetector(230, 190, 130,  // color
		                             45, true); // Lab threshold
	cv::namedWindow("result (functor)");
	result = colordetector(image);
	cv::imshow("result (functor)",result);

	// testing floodfill
	cv::floodFill(image,            // input/ouput image
		cv::Point(100, 50),         // seed point
		cv::Scalar(255, 255, 255),  // repainted color
		(cv::Rect*)0,  // bounding rectangle of the repainted pixel set
		cv::Scalar(35, 35, 35),     // low and high difference threshold
		cv::Scalar(35, 35, 35),     // most of the time will be identical
		cv::FLOODFILL_FIXED_RANGE); // pixels are compared to seed color

	cv::namedWindow("Flood Fill result");
	result = colordetector(image);
	cv::imshow("Flood Fill result", image);

	// Creating artificial images to demonstrate color space properties
	cv::Mat colors(100, 300, CV_8UC3, cv::Scalar(100, 200, 150));
	cv::Mat range= colors.colRange(0, 100);
	range = range + cv::Scalar(10, 10, 10);
	range = colors.colRange(200, 300);
	range = range + cv::Scalar(-10, -10, 10);

	cv::namedWindow("3 colors");
	cv::imshow("3 colors", colors);

	cv::Mat labImage(100, 300, CV_8UC3, cv::Scalar(100, 200, 150));
	cv::cvtColor(labImage, labImage, CV_BGR2Lab);
	range = colors.colRange(0, 100);
	range = range + cv::Scalar(10, 10, 10);
	range = colors.colRange(200, 300);
	range = range + cv::Scalar(-10, -10, 10);
	cv::cvtColor(labImage, labImage, CV_Lab2BGR);

	cv::namedWindow("3 colors (Lab)");
	cv::imshow("3 colors (Lab)", colors);

	// brightness versus luminance
	cv::Mat grayLevels(100, 256, CV_8UC3);
	for (int i = 0; i < 256; i++) {
		grayLevels.col(i) = cv::Scalar(i, i, i);
	}

	range = grayLevels.rowRange(50, 100);
	cv::Mat channels[3];
	cv::split(range, channels);
	channels[1] = 128;
	channels[2] = 128;
	cv::merge(channels, 3, range);
	cv::cvtColor(range, range, CV_Lab2BGR);


	cv::namedWindow("Luminance vs Brightness");
	cv::imshow("Luminance vs Brightness", grayLevels);

	cv::waitKey();

	return 0;
}

