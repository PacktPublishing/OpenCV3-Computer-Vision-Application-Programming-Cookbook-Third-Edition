/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 12 of the book:
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

#if !defined BGFGSeg
#define BGFGSeg

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoprocessor.h"

class BGFGSegmentor : public FrameProcessor {
	
	cv::Mat gray;			// current gray-level image
	cv::Mat background;		// accumulated background
	cv::Mat backImage;		// current background image
	cv::Mat foreground;		// foreground image
	double learningRate;    // learning rate in background accumulation
	int threshold;			// threshold for foreground extraction

  public:

	BGFGSegmentor() : threshold(10), learningRate(0.01) {}

	// Set the threshold used to declare a foreground
	void setThreshold(int t) {

		threshold= t;
	}

	// Set the learning rate
	void setLearningRate(double r) {

		learningRate= r;
	}

	// processing method
	void process(cv:: Mat &frame, cv:: Mat &output) {

		// convert to gray-level image
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); 

		// initialize background to 1st frame
		if (background.empty())
			gray.convertTo(background, CV_32F);
		
		// convert background to 8U
		background.convertTo(backImage,CV_8U);

		// compute difference between current image and background
		cv::absdiff(backImage,gray,foreground);

		// apply threshold to foreground image
		cv::threshold(foreground,output,threshold,255,cv::THRESH_BINARY_INV);

		// accumulate background
		cv::accumulateWeighted(gray, background, 
			                   // alpha*gray + (1-alpha)*background
			                   learningRate,  // alpha 
							   output);       // mask
	}
};

#endif
