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
#include <opencv2/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>


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

	// vector of keypoints and descriptors
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	cv::Mat descriptors1, descriptors2;

	// Construction of the SIFT feature detector 
	cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(74);

	// Detection of the SURF features
	ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	
	std::cout << "Number of SIFT points (1): " << keypoints1.size() << std::endl;
	std::cout << "Number of SIFT points (2): " << keypoints2.size() << std::endl;
	
	// Draw the kepoints
	cv::Mat imageKP;
	cv::drawKeypoints(image1,keypoints1,imageKP,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("Right SIFT Features");
	cv::imshow("Right SIFT Features",imageKP);
	cv::drawKeypoints(image2,keypoints2,imageKP,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("Left SIFT Features");
	cv::imshow("Left SIFT Features",imageKP);

	// Construction of the matcher 
	cv::BFMatcher matcher(cv::NORM_L2,true);

	// Match the two image descriptors
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1,descriptors2, matches);

	std::cout << "Number of matched points: " << matches.size() << std::endl;

	// Manually select few Matches  
	std::vector<cv::DMatch> selMatches;

	// make sure to double-check if the selected matches are valid
	selMatches.push_back(matches[2]);  
	selMatches.push_back(matches[5]);
	selMatches.push_back(matches[16]);
	selMatches.push_back(matches[19]);
	selMatches.push_back(matches[14]);
	selMatches.push_back(matches[34]);
	selMatches.push_back(matches[29]);

	// Draw the selected matches
	cv::Mat imageMatches;
	cv::drawMatches(image1,keypoints1,  // 1st image and its keypoints
		            image2,keypoints2,  // 2nd image and its keypoints
					selMatches,			// the selected matches
//					matches,			// the matches
					imageMatches,		// the image produced
					cv::Scalar(255,255,255),
					cv::Scalar(255,255,255),
					std::vector<char>(),
					2
					); // color of the lines
	cv::namedWindow("Matches");
	cv::imshow("Matches",imageMatches);
		
	// Convert 1 vector of keypoints into
	// 2 vectors of Point2f
	std::vector<int> pointIndexes1;
	std::vector<int> pointIndexes2;
	for (std::vector<cv::DMatch>::const_iterator it= selMatches.begin();
		 it!= selMatches.end(); ++it) {

			 // Get the indexes of the selected matched keypoints
			 pointIndexes1.push_back(it->queryIdx);
			 pointIndexes2.push_back(it->trainIdx);
	}
		 
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> selPoints1, selPoints2;
	cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
	cv::KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);

	// check by drawing the points 
	std::vector<cv::Point2f>::const_iterator it= selPoints1.begin();
	while (it!=selPoints1.end()) {

		// draw a circle at each corner location
		cv::circle(image1,*it,3,cv::Scalar(255,255,255),2);
		++it;
	}

	it= selPoints2.begin();
	while (it!=selPoints2.end()) {

		// draw a circle at each corner location
		cv::circle(image2,*it,3,cv::Scalar(255,255,255),2);
		++it;
	}

	// Compute F matrix from 7 matches
	cv::Mat fundamental= cv::findFundamentalMat(
		selPoints1, // points in first image
		selPoints2, // points in second image
		cv::FM_7POINT);       // 7-point method

	std::cout << "F-Matrix size= " << fundamental.rows << "," << fundamental.cols << std::endl;
	cv::Mat fund(fundamental, cv::Rect(0, 0, 3, 3));
	// draw the left points corresponding epipolar lines in right image 
	std::vector<cv::Vec3f> lines1; 
	cv::computeCorrespondEpilines(
		selPoints1, // image points 
		1,                   // in image 1 (can also be 2)
		fund, // F matrix
		lines1);     // vector of epipolar lines

	std::cout << "size of F matrix:" << fund.rows << "x" << fund.cols << std::endl;

	// for all epipolar lines
	for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
		 it!=lines1.end(); ++it) {

			 // draw the epipolar line between first and last column
			 cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
				             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
							 cv::Scalar(255,255,255));
	}
		
	// draw the left points corresponding epipolar lines in left image 
	std::vector<cv::Vec3f> lines2;
	cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fund,lines2);
	for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
		 it!=lines2.end(); ++it) {

			 // draw the epipolar line between first and last column
			 cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
				             cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
							 cv::Scalar(255,255,255));
	}
	
	// combine both images
	cv::Mat both(image1.rows,image1.cols+image2.cols, CV_8U);
	image1.copyTo(both.colRange(0, image1.cols));
	image2.copyTo(both.colRange(image1.cols, image1.cols+image2.cols));

    // Display the images with points and epipolar lines
	cv::namedWindow("Epilines");
	cv::imshow("Epilines",both);
/*
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2, newPoints1, newPoints2;
	cv::KeyPoint::convert(keypoints1, points1);
	cv::KeyPoint::convert(keypoints2, points2);
	cv::correctMatches(fund, points1, points2, newPoints1, newPoints2);
	cv::KeyPoint::convert(newPoints1, keypoints1);
	cv::KeyPoint::convert(newPoints2, keypoints2);

	cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
		image2, keypoints2,  // 2nd image and its keypoints
		matches,			// the matches
		imageMatches,		// the image produced
		cv::Scalar(255, 255, 255),
		cv::Scalar(255, 255, 255),
		std::vector<char>(),
		2
		); // color of the lines
	cv::namedWindow("Corrected matches");
	cv::imshow("Corrected matches", imageMatches);
	*/
	cv::waitKey();
	return 0;
}