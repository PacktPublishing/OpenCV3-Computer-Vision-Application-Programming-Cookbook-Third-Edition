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

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

int main()
{
	// open the positive sample images
	std::vector<cv::Mat> referenceImages;
	referenceImages.push_back(cv::imread("stopSamples/stop00.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop01.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop02.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop03.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop04.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop05.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop06.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop07.png"));

	// create a composite image
	cv::Mat positveImages(2 * referenceImages[0].rows, 4 * referenceImages[0].cols, CV_8UC3);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++) {

			referenceImages[i * 2 + j].copyTo(positveImages(cv::Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
		}

	cv::imshow("Positive samples", positveImages);

	cv::Mat negative = cv::imread("stopSamples/bg01.jpg");
	cv::resize(negative, negative, cv::Size(), 0.33, 0.33);
	cv::imshow("One negative sample", negative);

	cv::Mat inputImage = cv::imread("stopSamples/stop9.jpg");
	cv::resize(inputImage, inputImage, cv::Size(), 0.5, 0.5);
		
	cv::CascadeClassifier cascade;
	if (!cascade.load("stopSamples/classifier/cascade.xml")) { 
		std::cout << "Error when loading the cascade classfier!" << std::endl; 
		return -1; 
	}

	// predict the label of this image
	std::vector<cv::Rect> detections;

	cascade.detectMultiScale(inputImage, // input image 
		                     detections, // detection results
		                     1.1,        // scale reduction factor
		                     1,          // number of required neighbor detections
		                     0,          // flags (not used)
		                     cv::Size(48, 48),    // minimum object size to be detected
		                     cv::Size(128, 128)); // maximum object size to be detected

	std::cout << "detections= " << detections.size() << std::endl;
	for (int i = 0; i < detections.size(); i++)
		cv::rectangle(inputImage, detections[i], cv::Scalar(255, 255, 255), 2);

	cv::imshow("Stop sign detection", inputImage);

	// Detecting faces
	cv::Mat picture = cv::imread("girl.jpg");
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
		std::cout << "Error when loading the face cascade classfier!" << std::endl;
		return -1;
	}

	faceCascade.detectMultiScale(picture, // input image 
		detections, // detection results
		1.1,        // scale reduction factor
		3,          // number of required neighbor detections
		0,          // flags (not used)
		cv::Size(48, 48),    // minimum object size to be detected
		cv::Size(128, 128)); // maximum object size to be detected

	std::cout << "detections= " << detections.size() << std::endl;
	// draw detections on image
	for (int i = 0; i < detections.size(); i++)
		cv::rectangle(picture, detections[i], cv::Scalar(255, 255, 255), 2);

	// Detecting eyes
	cv::CascadeClassifier eyeCascade;
	if (!eyeCascade.load("haarcascade_eye.xml")) {
		std::cout << "Error when loading the eye cascade classfier!" << std::endl;
		return -1;
	}

	eyeCascade.detectMultiScale(picture, // input image 
		detections, // detection results
		1.1,        // scale reduction factor
		3,          // number of required neighbor detections
		0,          // flags (not used)
		cv::Size(24, 24),    // minimum object size to be detected
		cv::Size(64, 64)); // maximum object size to be detected

	std::cout << "detections= " << detections.size() << std::endl;
	// draw detections on image
	for (int i = 0; i < detections.size(); i++)
		cv::rectangle(picture, detections[i], cv::Scalar(0, 0, 0), 2);

	cv::imshow("Detection results", picture);

	cv::waitKey();
}