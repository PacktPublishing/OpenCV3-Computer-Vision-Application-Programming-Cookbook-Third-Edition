/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 11 of the book:
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
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/viz.hpp>
#include "robustMatcher.h"

int main()
{
	// Read input images
	cv::Mat image1= cv::imread("brebeuf1.jpg",0);
	cv::Mat image2= cv::imread("brebeuf2.jpg",0);
	if (!image1.data || !image2.data)
		return 0; 

	// Prepare the matcher (with default parameters)
	// here SIFT detector and descriptor
	RobustMatcher rmatcher(cv::xfeatures2d::SIFT::create(250));

	// Match the two images
	std::vector<cv::DMatch> matches;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat fundamental = rmatcher.match(image1, image2, matches,
		keypoints1, keypoints2);

	// draw the matches
	cv::Mat imageMatches;
	cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
		image2, keypoints2,  // 2nd image and its keypoints
		matches,			// the matches
		imageMatches,		// the image produced
		cv::Scalar(255, 255, 255),  // color of the lines
		cv::Scalar(255, 255, 255),  // color of the keypoints
		std::vector<char>(),
		2);
	cv::namedWindow("Matches");
	cv::imshow("Matches", imageMatches);

	// Convert keypoints into Point2f	
	std::vector<cv::Point2f> points1, points2;

	for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
	it != matches.end(); ++it) {

		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(keypoints1[it->queryIdx].pt);
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(keypoints2[it->trainIdx].pt);
	}

	// Compute homographic rectification
	cv::Mat h1, h2;
	cv::stereoRectifyUncalibrated(points1, points2, fundamental, image1.size(), h1, h2);

	// Rectify the images through warping
	cv::Mat rectified1;
	cv::warpPerspective(image1, rectified1, h1, image1.size());
	cv::Mat rectified2;
	cv::warpPerspective(image2, rectified2, h2, image1.size());
	// Display the images
	cv::namedWindow("Left Rectified Image");
	cv::imshow("Left Rectified Image", rectified1);
	cv::namedWindow("Right Rectified Image");
	cv::imshow("Right Rectified Image", rectified2);

	points1.clear();
	points2.clear();
	for (int i = 20; i < image1.rows - 20; i += 20) {

		points1.push_back(cv::Point(image1.cols / 2, i));
		points2.push_back(cv::Point(image2.cols / 2, i));
	}

	// Draw the epipolar lines
	std::vector<cv::Vec3f> lines1;
	cv::computeCorrespondEpilines(points1, 1, fundamental, lines1);

	for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin();
	it != lines1.end(); ++it) {

		cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}

	std::vector<cv::Vec3f> lines2;
	cv::computeCorrespondEpilines(points2, 2, fundamental, lines2);

	for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin();
	it != lines2.end(); ++it) {

		cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}

	// Display the images with epipolar lines
	cv::namedWindow("Left Epilines");
	cv::imshow("Left Epilines", image1);
	cv::namedWindow("Right Epilines");
	cv::imshow("Right Epilines", image2);

	// draw the pair
	cv::drawMatches(image1, keypoints1,  // 1st image 
		image2, keypoints2,              // 2nd image 
		std::vector<cv::DMatch>(),			
		imageMatches,		             // the image produced
		cv::Scalar(255, 255, 255),  
		cv::Scalar(255, 255, 255),  
		std::vector<char>(),
		2);
	cv::namedWindow("A Stereo pair");
	cv::imshow("A Stereo pair", imageMatches);

	// Compute disparity
	cv::Mat disparity;
	cv::Ptr<cv::StereoMatcher> pStereo = cv::StereoSGBM::create(0,   // minimum disparity
		                                                        32,  // maximum disparity
		                                                        5);  // block size
	pStereo->compute(rectified1, rectified2, disparity);

	// draw the rectified pair
	/*
	cv::warpPerspective(image1, rectified1, h1, image1.size());
	cv::warpPerspective(image2, rectified2, h2, image1.size());
	cv::drawMatches(rectified1, keypoints1,  // 1st image 
		rectified2, keypoints2,              // 2nd image
		std::vector<cv::DMatch>(),		
		imageMatches,		                // the image produced
		cv::Scalar(255, 255, 255),  
		cv::Scalar(255, 255, 255),  
		std::vector<char>(),
		2);
	cv::namedWindow("Rectified Stereo pair");
	cv::imshow("Rectified Stereo pair", imageMatches);
	*/

	double minv, maxv;
	disparity = disparity * 64;
	cv::minMaxLoc(disparity, &minv, &maxv);
	std::cout << minv << "+" << maxv << std::endl;
	// Display the disparity map
	cv::namedWindow("Disparity Map");
	cv::imshow("Disparity Map", disparity);

	cv::waitKey();
	return 0;
}