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
#include "robustMatcher.h"

int main()
{
	// Read input images
	cv::Mat image1= cv::imread("church01.jpg",0);
	cv::Mat image2= cv::imread("church03.jpg",0);

	if (!image1.data || !image2.data)
		return 0; 

    // Display the images
	cv::namedWindow("Right Image");
	cv::imshow("Right Image",image1);
	cv::namedWindow("Left Image");
	cv::imshow("Left Image",image2);

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
	cv::drawMatches(image1,keypoints1,  // 1st image and its keypoints
		            image2,keypoints2,  // 2nd image and its keypoints
					matches,			// the matches
					imageMatches,		// the image produced
					cv::Scalar(255,255,255),  // color of the lines
					cv::Scalar(255,255,255),  // color of the keypoints
					std::vector<char>(),
					2); 
	cv::namedWindow("Matches");
	cv::imshow("Matches",imageMatches);
	
	// Convert keypoints into Point2f	
	std::vector<cv::Point2f> points1, points2;
	
	for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
			 it!= matches.end(); ++it) {

			 // Get the position of left keypoints
			 float x= keypoints1[it->queryIdx].pt.x;
			 float y= keypoints1[it->queryIdx].pt.y;
			 points1.push_back(keypoints1[it->queryIdx].pt);
			 cv::circle(image1,cv::Point(x,y),3,cv::Scalar(255,255,255),3);
			 // Get the position of right keypoints
			 x= keypoints2[it->trainIdx].pt.x;
			 y= keypoints2[it->trainIdx].pt.y;
			 cv::circle(image2,cv::Point(x,y),3,cv::Scalar(255,255,255),3);
			 points2.push_back(keypoints2[it->trainIdx].pt);
	}
	
	// Draw the epipolar lines
	std::vector<cv::Vec3f> lines1; 
	cv::computeCorrespondEpilines(points1,1,fundamental,lines1);
		
	for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
			 it!=lines1.end(); ++it) {

			 cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
				             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
							 cv::Scalar(255,255,255));
	}

	std::vector<cv::Vec3f> lines2; 
	cv::computeCorrespondEpilines(points2,2,fundamental,lines2);
	
	for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
		     it!=lines2.end(); ++it) {

			 cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
				             cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
							 cv::Scalar(255,255,255));
	}

    // Display the images with epipolar lines
	cv::namedWindow("Right Image Epilines (RANSAC)");
	cv::imshow("Right Image Epilines (RANSAC)",image1);
	cv::namedWindow("Left Image Epilines (RANSAC)");
	cv::imshow("Left Image Epilines (RANSAC)",image2);

	cv::waitKey();
	return 0;
}