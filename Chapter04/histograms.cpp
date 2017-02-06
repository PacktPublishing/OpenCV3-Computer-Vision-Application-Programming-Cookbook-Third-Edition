/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 4 of the book:
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
using namespace std;

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "histogram.h"

int main()
{
	// Read input image
	cv::Mat image= cv::imread("group.jpg",0);
	if (!image.data)
		return 0; 

	// save grayscale image
	cv::imwrite("groupBW.jpg", image);

    // Display the image
	cv::namedWindow("Image");
	cv::imshow("Image",image);

	// The histogram object
	Histogram1D h;

    // Compute the histogram
	cv::Mat histo= h.getHistogram(image);

	// Loop over each bin
	for (int i=0; i<256; i++) 
		cout << "Value " << i << " = " << histo.at<float>(i) << endl;  

	// Display a histogram as an image
	cv::namedWindow("Histogram");
	cv::imshow("Histogram",h.getHistogramImage(image));

	// re-display the histagram with chosen threshold indicated
	cv::Mat hi = h.getHistogramImage(image);
	cv::line(hi, cv::Point(70, 0), cv::Point(70, 255), cv::Scalar(128));
	cv::namedWindow("Histogram with threshold value");
	cv::imshow("Histogram with threshold value", hi);

	// creating a binary image by thresholding at the valley
	cv::Mat thresholded; // output binary image
	cv::threshold(image,thresholded,
		          70,    // threshold value
				  255,   // value assigned to pixels over threshold value
				  cv::THRESH_BINARY); // thresholding type
 
	// Display the thresholded image
	cv::namedWindow("Binary Image");
	cv::imshow("Binary Image",thresholded);
	thresholded = 255 - thresholded;
	cv::imwrite("binary.bmp",thresholded);

	// Equalize the image
	cv::Mat eq= h.equalize(image);

	// Show the result
	cv::namedWindow("Equalized Image");
	cv::imshow("Equalized Image",eq);

	// Show the new histogram
	cv::namedWindow("Equalized H");
	cv::imshow("Equalized H",h.getHistogramImage(eq));

	// Stretch the image, setting the 1% of pixels at black and 1% at white
	cv::Mat str= h.stretch(image,0.01f);

	// Show the result
	cv::namedWindow("Stretched Image");
	cv::imshow("Stretched Image",str);

	// Show the new histogram
	cv::namedWindow("Stretched H");
	cv::imshow("Stretched H",h.getHistogramImage(str));

	// Create an image inversion table
	cv::Mat lut(1,256,CV_8U); // 1x256 matrix

	// Or:
	// int dim(256);
	// cv::Mat lut(1,  // 1 dimension
	// 	&dim,          // 256 entries
	//	CV_8U);        // uchar

	for (int i=0; i<256; i++) {
		
		// 0 becomes 255, 1 becomes 254, etc.
		lut.at<uchar>(i)= 255-i;
	}

	// Apply lookup and display negative image
	cv::namedWindow("Negative image");
	cv::imshow("Negative image",h.applyLookUp(image,lut));

	cv::waitKey();
	return 0;
}

