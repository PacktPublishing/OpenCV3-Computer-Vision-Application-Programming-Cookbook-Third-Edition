/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 13 of the book:
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


#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "videoprocessor.h"

// Drawing optical flow vectors on an image
void drawOpticalFlow(const cv::Mat& oflow,    // the optical flow 
	                 cv::Mat& flowImage,      // the produced image
	                 int stride,  // the stride for displaying the vectors
	                 float scale, // multiplying factor for the vectors
	                 const cv::Scalar& color) // the color of the vectors
{

	// create the image if required
	if (flowImage.size() != oflow.size()) {
		flowImage.create(oflow.size(), CV_8UC3);
		flowImage = cv::Vec3i(255,255,255);
	}

	// for all vectors using stride as a step
	for (int y = 0; y < oflow.rows; y += stride)
		for (int x = 0; x < oflow.cols; x += stride) {
			// gets the vector	
			cv::Point2f vector = oflow.at< cv::Point2f>(y, x);
			// draw the line	
			cv::line(flowImage, cv::Point(x, y), 
				     cv::Point(static_cast<int>(x + scale*vector.x + 0.5), 
						       static_cast<int>(y + scale*vector.y + 0.5)), color);
			// draw the arrow tip	
			cv::circle(flowImage, cv::Point(static_cast<int>(x + scale*vector.x + 0.5),
				                            static_cast<int>(y + scale*vector.y + 0.5)), 1, color, -1);
		}
}

int main()
{
	// pick 2 frames of the sequence
	cv::Mat frame1= cv::imread("goose/goose230.bmp", 0);
	cv::Mat frame2= cv::imread("goose/goose237.bmp", 0);

	// Combined display
	cv::Mat combined(frame1.rows, frame1.cols + frame2.cols, CV_8U);
	frame1.copyTo(combined.colRange(0, frame1.cols));
	frame2.copyTo(combined.colRange(frame1.cols, frame1.cols+frame2.cols));
	cv::imshow("Frames", combined);

	// Create the optical flow algorithm
	cv::Ptr<cv::DualTVL1OpticalFlow> tvl1 = cv::createOptFlow_DualTVL1();

	std::cout << "regularization coeeficient: " << tvl1->getLambda() << std::endl; // the smaller the soomther
	std::cout << "Number of scales: " << tvl1->getScalesNumber() << std::endl; // number of scales
	std::cout << "Scale step: " << tvl1->getScaleStep() << std::endl; // size between scales
	std::cout << "Number of warpings: " << tvl1->getWarpingsNumber() << std::endl; // size between scales
	std::cout << "Stopping criteria: " << tvl1->getEpsilon() << " and " << tvl1->getOuterIterations() << std::endl; // size between scales
																													// compute the optical flow between 2 frames
	cv::Mat oflow; // image of 2D flow vectors
	// compute optical flow between frame1 and frame2
	tvl1->calc(frame1, frame2, oflow);

	// Draw the optical flow image
	cv::Mat flowImage;
	drawOpticalFlow(oflow,     // input flow vectors 
		flowImage, // image to be generated
		8,         // display vectors every 8 pixels
		2,         // multiply size of vectors by 2
		cv::Scalar(0, 0, 0)); // vector color

	cv::imshow("Optical Flow", flowImage);

	// compute a smoother optical flow between 2 frames
	tvl1->setLambda(0.075);
	tvl1->calc(frame1, frame2, oflow);

	// Draw the optical flow image
	cv::Mat flowImage2;
	drawOpticalFlow(oflow,     // input flow vectors 
		flowImage2, // image to be generated
		8,         // display vectors every 8 pixels
		2,         // multiply size of vectors by 2
		cv::Scalar(0, 0, 0)); // vector color

	cv::imshow("Smoother Optical Flow", flowImage2);
	cv::waitKey();
}