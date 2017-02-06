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

int main()
{
	// image matching

	// 1. Read input images
	cv::Mat image1= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image2= cv::imread("church02.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	// 2. Define keypoints vector
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	// 3. Define feature detector
	cv::Ptr<cv::FeatureDetector> ptrDetector;           // generic detector
	ptrDetector= cv::FastFeatureDetector::create(80);   // we select the FAST detector

	// 4. Keypoint detection
	ptrDetector->detect(image1,keypoints1);
	ptrDetector->detect(image2,keypoints2);

	std::cout << "Number of keypoints (image 1): " << keypoints1.size() << std::endl; 
	std::cout << "Number of keypoints (image 2): " << keypoints2.size() << std::endl; 

	// 5. Define a square neighborhood
	const int nsize(11); // size of the neighborhood
	cv::Rect neighborhood(0, 0, nsize, nsize); // 11x11
	cv::Mat patch1;
	cv::Mat patch2;

	// 6. For all keypoints in first image
	//    find best match in second image
	cv::Mat result;
	std::vector<cv::DMatch> matches;

	//for all keypoints in image 1
	for (int i=0; i<keypoints1.size(); i++) {
	
		// define image patch
		neighborhood.x = keypoints1[i].pt.x-nsize/2;
		neighborhood.y = keypoints1[i].pt.y-nsize/2;

		// if neighborhood of points outside image, then continue with next point
		if (neighborhood.x<0 || neighborhood.y<0 ||
			neighborhood.x+nsize >= image1.cols || neighborhood.y+nsize >= image1.rows)
			continue;

		//patch in image 1
		patch1 = image1(neighborhood);

		// reset best correlation value;
		cv::DMatch bestMatch;

		//for all keypoints in image 2
	    for (int j=0; j<keypoints2.size(); j++) {

			// define image patch
			neighborhood.x = keypoints2[j].pt.x-nsize/2;
			neighborhood.y = keypoints2[j].pt.y-nsize/2;

			// if neighborhood of points outside image, then continue with next point
			if (neighborhood.x<0 || neighborhood.y<0 ||
				neighborhood.x + nsize >= image2.cols || neighborhood.y + nsize >= image2.rows)
				continue;

			// patch in image 2
			patch2 = image2(neighborhood);

			// match the two patches
			cv::matchTemplate(patch1,patch2,result, cv::TM_SQDIFF);

			// check if it is a best match
			if (result.at<float>(0,0) < bestMatch.distance) {

				bestMatch.distance= result.at<float>(0,0);
				bestMatch.queryIdx= i;
				bestMatch.trainIdx= j;
			}
		}

		// add the best match
		matches.push_back(bestMatch);
	}

	std::cout << "Number of matches: " << matches.size() << std::endl; 

	// extract the 50 best matches
	std::nth_element(matches.begin(),matches.begin()+50,matches.end());
	matches.erase(matches.begin()+50,matches.end());

	std::cout << "Number of matches (after): " << matches.size() << std::endl; 

	// Draw the matching results
	cv::Mat matchImage;
	cv::drawMatches(image1,keypoints1, // first image
                    image2,keypoints2, // second image
                    matches,     // vector of matches
                    matchImage,  // produced image
	                cv::Scalar(255,255,255),  // line color
		  		    cv::Scalar(255,255,255)); // point color

    // Display the image of matches
	cv::namedWindow("Matches");
	cv::imshow("Matches",matchImage);

	// Match template

	// define a template
	cv::Mat target(image1,cv::Rect(80,105,30,30));
    // Display the template
	cv::namedWindow("Template");
	cv::imshow("Template",target);

	// define search region
	cv::Mat roi(image2, 
		// here top half of the image
		cv::Rect(0,0,image2.cols,image2.rows/2)); 
			
	// perform template matching
	cv::matchTemplate(
		roi,    // search region
		target, // template
		result, // result
		CV_TM_SQDIFF); // similarity measure

	// find most similar location
	double minVal, maxVal;
	cv::Point minPt, maxPt;
	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);

	// draw rectangle at most similar location
	// at minPt in this case
	cv::rectangle(roi, cv::Rect(minPt.x, minPt.y, target.cols , target.rows), 255);
	
    // Display the template
	cv::namedWindow("Best");
	cv::imshow("Best",image2);

	cv::waitKey();
	return 0;
}