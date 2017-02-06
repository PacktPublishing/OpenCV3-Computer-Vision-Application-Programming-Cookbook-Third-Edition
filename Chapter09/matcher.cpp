/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 9 of the book:
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
#include <opencv2/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>

int main()
{
	// image matching

	// 1. Read input images
	cv::Mat image1= cv::imread("church01.jpg",cv::IMREAD_GRAYSCALE);
	cv::Mat image2= cv::imread("church02.jpg",cv::IMREAD_GRAYSCALE);

	// 2. Define keypoints vector
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	// 3. Define feature detector
	// Construct the SURF feature detector object
	cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SURF::create(2000.0);
	// to test with SIFT instead of SURF 
    // cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(74);

	// 4. Keypoint detection
	// Detect the SURF features
	ptrFeature2D->detect(image1,keypoints1);
	ptrFeature2D->detect(image2,keypoints2);

	// Draw feature points
	cv::Mat featureImage;
	cv::drawKeypoints(image1,keypoints1,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SURF");
	cv::imshow("SURF",featureImage);

	std::cout << "Number of SURF keypoints (image 1): " << keypoints1.size() << std::endl; 
	std::cout << "Number of SURF keypoints (image 2): " << keypoints2.size() << std::endl; 

	// SURF includes both the detector and descriptor extractor

	// 5. Extract the descriptor
    cv::Mat descriptors1;
    cv::Mat descriptors2;
	ptrFeature2D->compute(image1,keypoints1,descriptors1);
	ptrFeature2D->compute(image2,keypoints2,descriptors2);

    // Construction of the matcher 
	cv::BFMatcher matcher(cv::NORM_L2);
	// to test with crosscheck (symmetry) test
	// note: must not be used in conjunction with ratio test
    // cv::BFMatcher matcher(cv::NORM_L2, true); // with crosscheck
	// Match the two image descriptors
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1,descriptors2, matches);

    // draw matches
    cv::Mat imageMatches;
    cv::drawMatches(
	   image1, keypoints1, // 1st image and its keypoints
	   image2, keypoints2, // 2nd image and its keypoints
	   matches,            // the matches
	   imageMatches,       // the image produced
	   cv::Scalar(255, 255, 255),  // color of lines
	   cv::Scalar(255, 255, 255),  // color of points
	   std::vector< char >(),      // masks if any 
	   cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the image of matches
	cv::namedWindow("SURF Matches");
	cv::imshow("SURF Matches",imageMatches);

	std::cout << "Number of matches: " << matches.size() << std::endl; 

	// perform the ratio test

	// find the best two matches of each keypoint
	std::vector<std::vector<cv::DMatch> > matches2;
	matcher.knnMatch(descriptors1, descriptors2, 
		             matches2,
		             2); // find the k (2) best matches
	matches.clear();

	// perform ratio test
	double ratioMax= 0.6;
    std::vector<std::vector<cv::DMatch> >::iterator it;
	for (it= matches2.begin(); it!= matches2.end(); ++it) {
		//   first best match/second best match
		if ((*it)[0].distance/(*it)[1].distance < ratioMax) {
			// it is an acceptable match
			matches.push_back((*it)[0]);

		}
	}
	// matches is the new match set

    cv::drawMatches(
     image1,keypoints1, // 1st image and its keypoints
     image2,keypoints2, // 2nd image and its keypoints
     matches,           // the matches
     imageMatches,      // the image produced
     cv::Scalar(255,255,255),  // color of lines
     cv::Scalar(255,255,255),  // color of points
     std::vector< char >(),    // masks if any 
	 cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	std::cout << "Number of matches (after ratio test): " << matches.size() << std::endl; 

    // Display the image of matches
	cv::namedWindow("SURF Matches (ratio test at 0.6)");
	cv::imshow("SURF Matches (ratio test at 0.6)",imageMatches);

	// radius match
	float maxDist = 0.3;
	matches2.clear();
	matcher.radiusMatch(descriptors1, descriptors2, matches2,
		                maxDist); // maximum acceptable distance
				                  // between the 2 descriptors
	cv::drawMatches(
		image1, keypoints1, // 1st image and its keypoints
		image2, keypoints2, // 2nd image and its keypoints
		matches2,          // the matches
		imageMatches,      // the image produced
		cv::Scalar(255, 255, 255),  // color of lines
		cv::Scalar(255, 255, 255),  // color of points
		std::vector<std::vector< char >>(),    // masks if any 
		cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	int nmatches = 0;
	for (int i = 0; i< matches2.size(); i++) nmatches += matches2[i].size();
	std::cout << "Number of matches (with max radius): " << nmatches << std::endl;

	// Display the image of matches
	cv::namedWindow("SURF Matches (with max radius)");
	cv::imshow("SURF Matches (with max radius)", imageMatches);

	// scale-invariance test

	// Read input images
	image1= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	image2= cv::imread("church03.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	std::cout << "Number of SIFT keypoints (image 1): " << keypoints1.size() << std::endl; 
	std::cout << "Number of SIFT keypoints (image 2): " << keypoints2.size() << std::endl; 

	// Extract the keypoints and descriptors
	ptrFeature2D = cv::xfeatures2d::SIFT::create();
	ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // Match the two image descriptors
    matcher.match(descriptors1,descriptors2, matches);

	// extract the 50 best matches
	std::nth_element(matches.begin(),matches.begin()+50,matches.end());
	matches.erase(matches.begin()+50,matches.end());

   // draw matches
	cv::drawMatches(
		image1, keypoints1, // 1st image and its keypoints
		image2, keypoints2, // 2nd image and its keypoints
		matches,            // the matches
		imageMatches,      // the image produced
		cv::Scalar(255, 255, 255),  // color of lines
		cv::Scalar(255, 255, 255), // color of points
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS| cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // Display the image of matches
	cv::namedWindow("Multi-scale SIFT Matches");
	cv::imshow("Multi-scale SIFT Matches",imageMatches);

	std::cout << "Number of matches: " << matches.size() << std::endl; 

    cv::waitKey();
    return 0;
}
