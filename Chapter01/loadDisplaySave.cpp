/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 1 of the book:
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
#include <opencv2/imgproc.hpp>


void onMouse( int event, int x, int y, int flags, void* param)	{
	
	cv::Mat *im= reinterpret_cast<cv::Mat*>(param);

    switch (event) {	// dispatch the event

		case cv::EVENT_LBUTTONDOWN: // mouse button down event

			// display pixel value at (x,y)
			std::cout << "at (" << x << "," << y << ") value is: " 
				      << static_cast<int>(im->at<uchar>(cv::Point(x,y))) << std::endl;
			break;
	}
}

int main() {

	cv::Mat image; // create an empty image
	std::cout << "This image is " << image.rows << " x " 
              << image.cols << std::endl;

	// read the input image as a gray-scale image
	image=  cv::imread("puppy.bmp", cv::IMREAD_GRAYSCALE); 

    if (image.empty()) {  // error handling
        // no image has been created...
		// possibly display an error message
		// and quit the application 
		std::cout << "Error reading image..." << std::endl;
		return 0;
	}

	std::cout << "This image is " << image.rows << " x " 
			  << image.cols << std::endl;
	std::cout << "This image has " 
              << image.channels() << " channel(s)" << std::endl; 

	// create image window named "My Image"
    cv::namedWindow("Original Image"); // define the window (optional)
	cv::imshow("Original Image", image); // show the image

	// set the mouse callback for this image
	cv::setMouseCallback("Original Image", onMouse, reinterpret_cast<void*>(&image));

	cv::Mat result; // we create another empty image
	cv::flip(image,result,1); // positive for horizontal
                          // 0 for vertical,                     
                          // negative for both

	cv::namedWindow("Output Image"); // the output window
	cv::imshow("Output Image", result);

	cv::waitKey(0); // 0 to indefinitely wait for a key pressed
                // specifying a positive value will wait for
                // the given amount of msec

	cv::imwrite("output.bmp", result); // save result
	
	// create another image window named
	cv::namedWindow("Drawing on an Image"); // define the window

	cv::circle(image,              // destination image 
		       cv::Point(155,110), // center coordinate
			   65,                 // radius  
			   0,                  // color (here black)
			   3);                 // thickness
	
	cv::putText(image,                   // destination image
		        "This is a dog.",        // text
				cv::Point(40,200),       // text position
				cv::FONT_HERSHEY_PLAIN,  // font type
				2.0,                     // font scale
				255,                     // text color (here white)
				2);                      // text thickness

	cv::imshow("Drawing on an Image", image); // show the image

	cv::waitKey(0); // 0 to indefinitely wait for a key pressed
	
	return 0;
}

