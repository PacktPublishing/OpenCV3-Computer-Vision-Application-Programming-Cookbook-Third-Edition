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

#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

int main()
{
	// Read input image
	cv::Mat image= cv::imread("binary.bmp");
	if (!image.data)
		return 0; 

    // Display the image
	cv::namedWindow("Image");
	cv::imshow("Image",image);

	// Erode the image
	// with the default 3x3 structuring element (SE)
	cv::Mat eroded; // the destination image
	cv::erode(image,eroded,cv::Mat());

    // Display the eroded image
	cv::namedWindow("Eroded Image");
	cv::imshow("Eroded Image",eroded);

	// Dilate the image
	cv::Mat dilated; // the destination image
	cv::dilate(image,dilated,cv::Mat());

    // Display the dilated image
	cv::namedWindow("Dilated Image");
	cv::imshow("Dilated Image",dilated);

	// Erode the image with a larger SE
	// create a 7x7 mat with containing all 1s
	cv::Mat element(7,7,CV_8U,cv::Scalar(1));
	// erode the image with that SE
	cv::erode(image,eroded,element);

    // Display the eroded image
	cv::namedWindow("Eroded Image (7x7)");
	cv::imshow("Eroded Image (7x7)",eroded);

	// Erode the image 3 times.
	cv::erode(image,eroded,cv::Mat(),cv::Point(-1,-1),3);

    // Display the eroded image
	cv::namedWindow("Eroded Image (3 times)");
	cv::imshow("Eroded Image (3 times)",eroded);

	// Close the image
	cv::Mat element5(5,5,CV_8U,cv::Scalar(1));
	cv::Mat closed;
	cv::morphologyEx(image,closed,    // input and output images
		             cv::MORPH_CLOSE, // operator code
		             element5);       // structuring element

    // Display the closed image
	cv::namedWindow("Closed Image");
	cv::imshow("Closed Image",closed);

	// Open the image
	cv::Mat opened;
	cv::morphologyEx(image,opened,cv::MORPH_OPEN,element5);

    // Display the opened image
	cv::namedWindow("Opened Image");
	cv::imshow("Opened Image",opened);

	// explicit closing
	// 1. dilate original image
	cv::Mat result;
	cv::dilate(image, result, element5);
	// 2. in-place erosion of the dilated image
	cv::erode(result, result, element5);

	// Display the closed image
	cv::namedWindow("Closed Image (2)");
	cv::imshow("Closed Image (2)", result);

	// Close and Open the image
	cv::morphologyEx(image,image,cv::MORPH_CLOSE,element5);
	cv::morphologyEx(image,image,cv::MORPH_OPEN,element5);

    // Display the close/opened image
	cv::namedWindow("Closed|Opened Image");
	cv::imshow("Closed|Opened Image",image);
	cv::imwrite("binaryGroup.bmp",image);

	// Read input image
	image= cv::imread("binary.bmp");

	// Open and Close the image
	cv::morphologyEx(image,image,cv::MORPH_OPEN,element5);
	cv::morphologyEx(image,image,cv::MORPH_CLOSE,element5);

    // Display the close/opened image
	cv::namedWindow("Opened|Closed Image");
	cv::imshow("Opened|Closed Image",image);

	// Read input image (gray-level)
	image = cv::imread("boldt.jpg",0);
	if (!image.data)
		return 0;

	// Get the gradient image using a 3x3 structuring element
	cv::morphologyEx(image, result, cv::MORPH_GRADIENT, cv::Mat());

	// Display the morphological edge image
	cv::namedWindow("Edge Image");
	cv::imshow("Edge Image", 255 - result);

	// Apply threshold to obtain a binary image
	int threshold(80);
	cv::threshold(result, result,
					threshold, 255, cv::THRESH_BINARY);

	// Display the close/opened image
	cv::namedWindow("Thresholded Edge Image");
	cv::imshow("Thresholded Edge Image", result);

	// Get the gradient image using a 3x3 structuring element
	cv::morphologyEx(image, result, cv::MORPH_GRADIENT, cv::Mat());

	// Read input image (gray-level)
	image = cv::imread("book.jpg", 0);
	if (!image.data)
		return 0;
	// rotate the image for easier display
	cv::transpose(image, image);
	cv::flip(image, image, 0);

	// Apply the black top-hat transform using a 7x7 structuring element
	cv::Mat element7(7, 7, CV_8U, cv::Scalar(1));
	cv::morphologyEx(image, result, cv::MORPH_BLACKHAT, element7);

	// Display the top-hat image
	cv::namedWindow("7x7 Black Top-hat Image");
	cv::imshow("7x7 Black Top-hat Image", 255-result);

	// Apply threshold to obtain a binary image
	threshold= 25;
	cv::threshold(result, result,
		threshold, 255, cv::THRESH_BINARY);

	// Display the morphological edge image
	cv::namedWindow("Thresholded Black Top-hat");
	cv::imshow("Thresholded Black Top-hat", 255 - result);

	// Apply the black top-hat transform using a 7x7 structuring element
	cv::morphologyEx(image, result, cv::MORPH_CLOSE, element7);

	// Display the top-hat image
	cv::namedWindow("7x7 Closed Image");
	cv::imshow("7x7 Closed Image", 255 - result);

	cv::waitKey();
	return 0;
}

