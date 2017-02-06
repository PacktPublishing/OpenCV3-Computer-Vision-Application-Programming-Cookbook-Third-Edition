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
#include <opencv2/stitching.hpp>

int main()
{
	// Read input images
	cv::Mat image1= cv::imread("parliament1.jpg",0);
	cv::Mat image2= cv::imread("parliament2.jpg",0);
	if (!image1.data || !image2.data)
		return 0; 

    // Display the images
	cv::namedWindow("Image 1");
	cv::imshow("Image 1",image1);
	cv::namedWindow("Image 2");
	cv::imshow("Image 2",image2);
	
	// vector of keypoints and descriptors
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	cv::Mat descriptors1, descriptors2;

	// 1. Construction of the SIFT feature detector 
	cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(74);

	// 2. Detection of the SIFT features and associated descriptors
	ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

	std::cout << "Number of feature points (1): " << keypoints1.size() << std::endl;
	std::cout << "Number of feature points (2): " << keypoints2.size() << std::endl;

	// 3. Match the two image descriptors
   
	// Construction of the matcher with crosscheck 
	cv::BFMatcher matcher(cv::NORM_L2, true);                            
	// matching
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1,descriptors2,matches);

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
	cv::namedWindow("Matches (pure rotation case)");
	cv::imshow("Matches (pure rotation case)",imageMatches);
	
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
		 it!= matches.end(); ++it) {

			 // Get the position of left keypoints
			 float x= keypoints1[it->queryIdx].pt.x;
			 float y= keypoints1[it->queryIdx].pt.y;
			 points1.push_back(cv::Point2f(x,y));
			 // Get the position of right keypoints
			 x= keypoints2[it->trainIdx].pt.x;
			 y= keypoints2[it->trainIdx].pt.y;
			 points2.push_back(cv::Point2f(x,y));
	}

	std::cout << points1.size() << " " << points2.size() << std::endl; 

	// Find the homography between image 1 and image 2
	std::vector<char> inliers;
	cv::Mat homography= cv::findHomography(
		points1,points2, // corresponding points
		inliers,	     // outputed inliers matches 
		cv::RANSAC,	     // RANSAC method
		1.);	         // max distance to reprojection point
	
    // Draw the inlier points
	cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
		image2, keypoints2,  // 2nd image and its keypoints
		matches,			// the matches
		imageMatches,		// the image produced
		cv::Scalar(255, 255, 255),  // color of the lines
		cv::Scalar(255, 255, 255),  // color of the keypoints
		inliers,
		2);
	cv::namedWindow("Homography inlier points");
	cv::imshow("Homography inlier points", imageMatches);

	// Warp image 1 to image 2
	cv::Mat result;
	cv::warpPerspective(image1, // input image
		result,			// output image
		homography,		// homography
		cv::Size(2*image1.cols,image1.rows)); // size of output image

	// Copy image 1 on the first half of full image
	cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
	image2.copyTo(half);

    // Display the warp image
	cv::namedWindow("Image mosaic");
	cv::imshow("Image mosaic",result);

	// Read input images
	std::vector<cv::Mat> images;
	images.push_back(cv::imread("parliament1.jpg"));
	images.push_back(cv::imread("parliament2.jpg"));

	cv::Mat panorama; // output panorama
	// create the stitcher
	cv::Stitcher stitcher = cv::Stitcher::createDefault();
	// stitch the images
	cv::Stitcher::Status status = stitcher.stitch(images, panorama);

	if (status == cv::Stitcher::OK) // success?
	{
		// Display the panorama
		cv::namedWindow("Panorama");
		cv::imshow("Panorama", panorama);
	}

	cv::waitKey();
	return 0;
}