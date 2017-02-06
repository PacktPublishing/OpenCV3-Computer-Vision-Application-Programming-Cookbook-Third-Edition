/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 8 of the book:
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
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "harrisDetector.h"

int main()
{
	// Harris:

	// Read input image
	cv::Mat image= cv::imread("church01.jpg",0);
	if (!image.data)
		return 0; 

	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

    // Display the image
	cv::namedWindow("Original");
	cv::imshow("Original",image);

	// Detect Harris corners
	cv::Mat cornerStrength;
	cv::cornerHarris(image, cornerStrength,
		             3,     // neighborhood size
					 3,     // aperture size
					 0.01); // Harris parameter

   // threshold the corner strengths
	cv::Mat harrisCorners;
	double threshold= 0.0001; 
	cv::threshold(cornerStrength, harrisCorners,
                  threshold,255,cv::THRESH_BINARY_INV);
	 
    // Display the corners
	cv::namedWindow("Harris");
	cv::imshow("Harris", harrisCorners);

	// Create Harris detector instance
	HarrisDetector harris;
    // Compute Harris values
	harris.detect(image);
    // Detect Harris corners
	std::vector<cv::Point> pts;
	harris.getCorners(pts,0.02);
	// Draw Harris corners
	harris.drawOnImage(image,pts);

    // Display the corners
	cv::namedWindow("Corners");
	cv::imshow("Corners",image);

	// GFTT:

	// Read input image
	image= cv::imread("church01.jpg",0);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	// Compute good features to track

	std::vector<cv::KeyPoint> keypoints;
	// GFTT detector
	cv::Ptr<cv::GFTTDetector> ptrGFTT = cv::GFTTDetector::create(
		500,	// maximum number of keypoints to be returned
		0.01,	// quality level
		10);	// minimum allowed distance between points	  
	// detect the GFTT
	ptrGFTT->detect(image,keypoints);
	// for all keypoints
	std::vector<cv::KeyPoint>::const_iterator it= keypoints.begin();
	while (it!=keypoints.end()) {

		// draw a circle at each corner location
		cv::circle(image,it->pt,3,cv::Scalar(255,255,255),1);
		++it;
	}

    // Display the keypoints
	cv::namedWindow("GFTT");
	cv::imshow("GFTT",image);

	// Read input image
	image= cv::imread("church01.jpg",0);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	// draw the keypoints
	cv::drawKeypoints(image,		// original image
		keypoints,					// vector of keypoints
		image,						// the resulting image
		cv::Scalar(255,255,255),	// color of the points
		cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag

    // Display the keypoints
	cv::namedWindow("Good Features to Track Detector");
	cv::imshow("Good Features to Track Detector",image);

	// FAST feature:

	// Read input image
	image= cv::imread("church01.jpg",0);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);
	keypoints.clear();
	// FAST detector
	cv::Ptr<cv::FastFeatureDetector> ptrFAST = cv::FastFeatureDetector::create(40);
	// detect the keypoints
	ptrFAST->detect(image,keypoints);
	// draw the keypoints
	cv::drawKeypoints(image,keypoints,image,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
	std::cout << "Number of keypoints (FAST): " << keypoints.size() << std::endl; 

    // Display the keypoints
	cv::namedWindow("FAST");
	cv::imshow("FAST",image);

	// FAST feature without non-max suppression
	// Read input image
	image= cv::imread("church01.jpg",0);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	keypoints.clear();
	// detect the keypoints
	ptrFAST->setNonmaxSuppression(false);

	ptrFAST->detect(image, keypoints);
	// draw the keypoints
	cv::drawKeypoints(image,keypoints,image,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

    // Display the keypoints
	cv::namedWindow("FAST Features (all)");
	cv::imshow("FAST Features (all)",image);
	
	// Read input image
	image= cv::imread("church01.jpg",0);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	int total(100); // requested number of keypoints
	int hstep(5), vstep(3); // a grid of 4 columns by 3 rows
	// hstep= vstep= 1; // try without grid
	int hsize(image.cols / hstep), vsize(image.rows / vstep);
	int subtotal(total / (hstep*vstep)); // number of keypoints per grid
	cv::Mat imageROI;
	std::vector<cv::KeyPoint> gridpoints;

	std::cout << "Grid of " << vstep << " by " << hstep << " each of size " << vsize << " by " << hsize << std::endl;

	// detection with low threshold
	ptrFAST->setThreshold(20); 
	// non-max suppression
	ptrFAST->setNonmaxSuppression(true);

	// The final vector of keypoints
	keypoints.clear();
	// detect on each grid
	for (int i = 0; i < vstep; i++)
		for (int j = 0; j < hstep; j++) {

			// create ROI over current grid
			imageROI = image(cv::Rect(j*hsize, i*vsize, hsize, vsize));
			// detect the keypoints in grid
			gridpoints.clear();
			ptrFAST->detect(imageROI, gridpoints);
			std::cout << "Number of FAST in grid " << i << "," << j << ": " << gridpoints.size() << std::endl;
			if (gridpoints.size() > subtotal) {
				for (auto it = gridpoints.begin(); it != gridpoints.begin() + subtotal; ++it) {
					std::cout << "  " << it->response << std::endl;
				}
			}

			// get the strongest FAST features
			auto itEnd(gridpoints.end());
			if (gridpoints.size() > subtotal) { // select the strongest features
				std::nth_element(gridpoints.begin(), gridpoints.begin() + subtotal, gridpoints.end(),
					             [](cv::KeyPoint& a, cv::KeyPoint& b) {return a.response > b.response; });
				itEnd = gridpoints.begin() + subtotal;
			}

			// add them to the global keypoint vector
			for (auto it = gridpoints.begin(); it != itEnd; ++it) {
				it->pt += cv::Point2f(j*hsize, i*vsize); // convert to image coordinates
				keypoints.push_back(*it);
				std::cout << "  " <<it->response << std::endl;
			}
		}

	// draw the keypoints
	cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

	// Display the keypoints
	cv::namedWindow("FAST Features (grid)");
	cv::imshow("FAST Features (grid)", image);

	// SURF:

	// Read input image
	image = cv::imread("church01.jpg", 0);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	keypoints.clear();
	// Construct the SURF feature detector object
	cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> ptrSURF = cv::xfeatures2d::SurfFeatureDetector::create(2000.0);
	// detect the keypoints
	ptrSURF->detect(image, keypoints);
	
	// Detect the SURF features
	ptrSURF->detect(image,keypoints);
	
	cv::Mat featureImage;
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the keypoints
	cv::namedWindow("SURF");
	cv::imshow("SURF",featureImage);

	std::cout << "Number of SURF keypoints: " << keypoints.size() << std::endl; 

	// Read a second input image
	image= cv::imread("church03.jpg", cv::IMREAD_GRAYSCALE);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	// Detect the SURF features
	ptrSURF->detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the keypoints
	cv::namedWindow("SURF (2)");
	cv::imshow("SURF (2)",featureImage);

	// SIFT:

	// Read input image
	image= cv::imread("church01.jpg", cv::IMREAD_GRAYSCALE);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	keypoints.clear();
	// Construct the SIFT feature detector object
	cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> ptrSIFT = cv::xfeatures2d::SiftFeatureDetector::create();
	// detect the keypoints
	ptrSIFT->detect(image, keypoints);

	// Detect the SIFT features
	ptrSIFT->detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the keypoints
	cv::namedWindow("SIFT");
	cv::imshow("SIFT",featureImage);

	std::cout << "Number of SIFT keypoints: " << keypoints.size() << std::endl; 

	// BRISK:

	// Read input image
	image= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

    // Display the keypoints
	cv::namedWindow("BRISK");
	cv::imshow("BRISK",featureImage);

	keypoints.clear();
	// Construct another BRISK feature detector object
	cv::Ptr<cv::BRISK> ptrBRISK = cv::BRISK::create(
		60,  // threshold for BRISK points to be accepted
		5);  // number of octaves

	// Detect the BRISK features
	ptrBRISK->detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Display the keypoints
	cv::namedWindow("BRISK");
	cv::imshow("BRISK", featureImage);

	std::cout << "Number of BRISK keypoints: " << keypoints.size() << std::endl; 

	// ORB:

	// Read input image
	image= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	// rotate the image (to produce a horizontal image)
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	keypoints.clear();
	// Construct the BRISK feature detector object
	cv::Ptr<cv::ORB> ptrORB = cv::ORB::create(
		75, // total number of keypoints
		1.2, // scale factor between layers
		8);  // number of layers in pyramid
	// detect the keypoints
	ptrORB->detect(image, keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the keypoints
	cv::namedWindow("ORB");
	cv::imshow("ORB",featureImage);

	std::cout << "Number of ORB keypoints: " << keypoints.size() << std::endl; 

	cv::waitKey();
	return 0;
}