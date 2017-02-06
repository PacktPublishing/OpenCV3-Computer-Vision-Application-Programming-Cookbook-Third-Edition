/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 1 of the book:
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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main() {

	// define an image window
	cv::namedWindow("Image"); 

	// read the image 
	cv::Mat image=  cv::imread("puppy.bmp"); 

	// read the logo
	cv::Mat logo=  cv::imread("smalllogo.png"); 

	// define image ROI at image bottom-right
	cv::Mat imageROI(image, 
		          cv::Rect(image.cols-logo.cols, //ROI coordinates
                           image.rows-logo.rows,
		                   logo.cols,logo.rows));// ROI size

	// insert logo
	logo.copyTo(imageROI);

	cv::imshow("Image", image); // show the image
	cv::waitKey(0); // wait for a key pressed

	// re-read the original image
	image=  cv::imread("puppy.bmp");

	// define image ROI at image bottom-right
	imageROI= image(cv::Rect(image.cols-logo.cols,image.rows-logo.rows,
		                     logo.cols,logo.rows));
	// or using ranges:
    // imageROI= image(cv::Range(image.rows-logo.rows,image.rows), 
    //                 cv::Range(image.cols-logo.cols,image.cols));

    // use the logo as a mask (must be gray-level)
    cv::Mat mask(logo);

	// insert by copying only at locations of non-zero mask
	logo.copyTo(imageROI,mask);

	cv::imshow("Image", image); // show the image
	cv::waitKey(0); // wait for a key pressed

    return 0;
}

