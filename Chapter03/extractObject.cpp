/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 3 of the book:
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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


int main()
{
	// Read input image
	cv::Mat image= cv::imread("boldt.jpg");
	if (!image.data)
		return 0; 

    // Display the image
	cv::namedWindow("Original Image");
	cv::imshow("Original Image",image);

	// define bounding rectangle 
	cv::Rect rectangle(50,25,210,180);
	// the models (internally used)
	cv::Mat bgModel,fgModel; 
	// segmentation result
	cv::Mat result; // segmentation (4 possible values)

	// GrabCut segmentation
	cv::grabCut(image,    // input image
		result,   // segmentation result
		rectangle,// rectangle containing foreground 
		bgModel, fgModel, // models
		5,        // number of iterations
		cv::GC_INIT_WITH_RECT); // use rectangle

	// Get the pixels marked as likely foreground
	cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
	// or:
    //	result= result&1;

	// create a white image
	cv::Mat foreground(image.size(), CV_8UC3,
	          	       cv::Scalar(255, 255, 255));

	image.copyTo(foreground,result); // bg pixels not copied

	// draw rectangle on original image
	cv::rectangle(image, rectangle, cv::Scalar(255,255,255),1);
	cv::namedWindow("Image with rectangle");
	cv::imshow("Image with rectangle",image);

	// display result
	cv::namedWindow("Foreground object");
	cv::imshow("Foreground object",foreground);

	cv::waitKey();
	return 0;
}
