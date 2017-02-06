/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 5 of the book:
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
#include <vector>

int main()
{
	// Read input image
	cv::Mat image= cv::imread("building.jpg",0);
	if (!image.data)
		return 0; 

    // Display the image
	cv::namedWindow("Image");
	cv::imshow("Image",image);
	

	// basic MSER detector
	cv::Ptr<cv::MSER> ptrMSER= cv::MSER::create(5,     // delta value for local minima detection
		                                        200,   // min acceptable area 
				                                2000); // max acceptable area

    // vector of point sets
	std::vector<std::vector<cv::Point> > points;
	// vector of rectangles
	std::vector<cv::Rect> rects;
	// detect MSER features
	ptrMSER->detectRegions(image, points, rects);

	std::cout << points.size() << " MSERs detected" << std::endl;

	// create white image
	cv::Mat output(image.size(),CV_8UC3);
	output= cv::Scalar(255,255,255);
	
	// OpenCV random number generator
	cv::RNG rng;

	// Display the MSERs in color areas
	// for each detected feature
	// reverse order to display the larger MSER first
    for (std::vector<std::vector<cv::Point> >::reverse_iterator it= points.rbegin();
			   it!= points.rend(); ++it) {

		// generate a random color
		cv::Vec3b c(rng.uniform(0,254),
			        rng.uniform(0,254),
					rng.uniform(0,254));

		std::cout << "MSER size= " << it->size() << std::endl;

		// for each point in MSER set
		for (std::vector<cv::Point>::iterator itPts= it->begin();
			                                  itPts!= it->end(); ++itPts) {

			//do not overwrite MSER pixels
			if (output.at<cv::Vec3b>(*itPts)[0]==255) {

				output.at<cv::Vec3b>(*itPts)= c;
			}
		}
	}
	
	cv::namedWindow("MSER point sets");
	cv::imshow("MSER point sets",output);
	cv::imwrite("mser.bmp", output);

	// Extract and display the rectangular MSERs
	std::vector<cv::Rect>::iterator itr = rects.begin();
	std::vector<std::vector<cv::Point> >::iterator itp = points.begin();
	for (; itr != rects.end(); ++itr, ++itp) {

		// ratio test
		if (static_cast<double>(itp->size())/itr->area() > 0.6)
			cv::rectangle(image, *itr, cv::Scalar(255), 2);
	}

	// Display the resulting image
	cv::namedWindow("Rectangular MSERs");
	cv::imshow("Rectangular MSERs", image);
	
	// Reload the input image
	image = cv::imread("building.jpg", 0);
	if (!image.data)
		return 0;

	// Extract and display the elliptic MSERs
	for (std::vector<std::vector<cv::Point> >::iterator it = points.begin();
	                                                    it != points.end(); ++it) {

		// for each point in MSER set
		for (std::vector<cv::Point>::iterator itPts = it->begin();
		                                      itPts != it->end(); ++itPts) {

			// Extract bouding rectangles
			cv::RotatedRect rr = cv::minAreaRect(*it);
            // check ellipse elongation
			if (rr.size.height / rr.size.height > 0.6 || rr.size.height / rr.size.height < 1.6)
				cv::ellipse(image, rr, cv::Scalar(255), 2);
		}
	}

	// Display the image
	cv::namedWindow("MSER ellipses");
	cv::imshow("MSER ellipses", image);

	/*
	// detection using mserFeatures class

	// create MSER feature detector instance
	MSERFeatures mserF(200,  // min area 
		               1500, // max area
					   0.5); // ratio area threshold
	                         // default delta is used

	// the vector of bounding rotated rectangles
	std::vector<cv::RotatedRect> rects;

	// detect and get the image
	cv::Mat result= mserF.getImageOfEllipses(image,rects);

	// display detected MSER
	cv::namedWindow("MSER regions");
	cv::imshow("MSER regions",result);
	*/
	cv::waitKey();
}
