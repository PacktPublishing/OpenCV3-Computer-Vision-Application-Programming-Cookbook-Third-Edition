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

#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "videoprocessor.h"

void draw(const cv::Mat& img, cv::Mat& out) {

	img.copyTo(out);
	cv::circle(out, cv::Point(100,100),5,cv::Scalar(255,0,0),2);
}

// processing function
void canny(cv::Mat& img, cv::Mat& out) {

   // Convert to gray
   if (img.channels()==3)
      cv::cvtColor(img,out,cv::COLOR_BGR2GRAY);
   // Compute Canny edges
   cv::Canny(out,out,100,200);
   // Invert the image
   cv::threshold(out,out,128,255,cv::THRESH_BINARY_INV);
}

int main()
{
	// Open the video file
	cv::VideoCapture capture("bike.avi");
//	cv::VideoCapture capture("http://www.laganiere.name/bike.avi");
	// check if video successfully opened
	if (!capture.isOpened())
		return 1;

	// Get the frame rate
	double rate= capture.get(cv::CAP_PROP_FPS);
	std::cout << "Frame rate: " << rate << "fps" << std::endl;

	bool stop(false);
	cv::Mat frame; // current video frame
	cv::namedWindow("Extracted Frame");

	// Delay between each frame
	// corresponds to video frame rate
	int delay= 1000/rate;
	long long i=0;
	std::string b="bike";
	std::string ext=".bmp";
	// for all frames in video
	while (!stop) {

		// read next frame if any
		if (!capture.read(frame))
			break;

		cv::imshow("Extracted Frame",frame);

		std::string name(b);
        std::ostringstream ss; ss << std::setfill('0') << std::setw(3) << i; name+= ss.str(); i++;
		name+=ext;

		std::cout << name <<std::endl;
		
		cv::Mat test;
		//      cv::resize(frame, test, cv::Size(), 0.2,0.2);
		//		cv::imwrite(name, frame);
        //		cv::imwrite(name, test);

		// introduce a delay
		// or press key to stop
		if (cv::waitKey(delay)>=0)
				stop= true;
	}

	// Close the video file
	capture.release();

	cv::waitKey();

	// Now using the VideoProcessor class

	// Create instance
	VideoProcessor processor;

	// Open video file
	processor.setInput("bike.avi");

	// Declare a window to display the video
	processor.displayInput("Input Video");
	processor.displayOutput("Output Video");

	// Play the video at the original frame rate
	processor.setDelay(1000./processor.getFrameRate());

	// Set the frame processor callback function
	processor.setFrameProcessor(canny);

	// output a video
	processor.setOutput("bikeCanny.avi",-1,15);

	// stop the process at this frame
	processor.stopAtFrameNo(51);

	// Start the process
	processor.run();

	cv::waitKey();	

	return 0;
}
