/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 5 of the book:
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
#include <opencv2/imgproc.hpp>
#include "watershedSegmentation.h"


int main()
{
	// Read input image
	cv::Mat image= cv::imread("group.jpg");
	if (!image.data)
		return 0; 

    // Display the image
	cv::namedWindow("Original Image");
	cv::imshow("Original Image",image);

	// Get the binary map
	cv::Mat binary;
	binary= cv::imread("binary.bmp",0);

    // Display the binary image
	cv::namedWindow("Binary Image");
	cv::imshow("Binary Image",binary);

	// Eliminate noise and smaller objects
	cv::Mat fg;
	cv::erode(binary,fg,cv::Mat(),cv::Point(-1,-1),4);

    // Display the foreground image
	cv::namedWindow("Foreground Image");
	cv::imshow("Foreground Image",fg);

	// Identify image pixels without objects
	cv::Mat bg;
	cv::dilate(binary,bg,cv::Mat(),cv::Point(-1,-1),4);
	cv::threshold(bg,bg,1,128,cv::THRESH_BINARY_INV);

    // Display the background image
	cv::namedWindow("Background Image");
	cv::imshow("Background Image",bg);

	// Show markers image
	cv::Mat markers(binary.size(),CV_8U,cv::Scalar(0));
	markers= fg+bg;
	cv::namedWindow("Markers");
	cv::imshow("Markers",markers);

	// Create watershed segmentation object
	WatershedSegmenter segmenter;

	// Set markers and process
	segmenter.setMarkers(markers);
    segmenter.process(image);

    // Display segmentation result
	cv::namedWindow("Segmentation");
	cv::imshow("Segmentation",segmenter.getSegmentation());

    // Display watersheds
	cv::namedWindow("Watersheds");
	cv::imshow("Watersheds",segmenter.getWatersheds());

    // Open another image
	image= cv::imread("tower.jpg");

	// Identify background pixels
	cv::Mat imageMask(image.size(),CV_8U,cv::Scalar(0));
	cv::rectangle(imageMask,cv::Point(5,5),cv::Point(image.cols-5,image.rows-5),cv::Scalar(255),3);
	// Identify foreground pixels (in the middle of the image)
	cv::rectangle(imageMask,cv::Point(image.cols/2-10,image.rows/2-10),
						    cv::Point(image.cols/2+10,image.rows/2+10),cv::Scalar(1),10);

	// Set markers and process
	segmenter.setMarkers(imageMask);
	segmenter.process(image);

    // Display the image with markers
	cv::rectangle(image,cv::Point(5,5),cv::Point(image.cols-5,image.rows-5),cv::Scalar(255,255,255),3);
	cv::rectangle(image,cv::Point(image.cols/2-10,image.rows/2-10),
						 cv::Point(image.cols/2+10,image.rows/2+10),cv::Scalar(1,1,1),10);
	cv::namedWindow("Image with marker");
	cv::imshow("Image with marker",image);

	// Display watersheds
	cv::namedWindow("Watershed");
	cv::imshow("Watershed",segmenter.getWatersheds());

	cv::waitKey();
	return 0;
}
