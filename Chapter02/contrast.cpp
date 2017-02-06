/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 2 of the book:
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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void sharpen(const cv::Mat &image, cv::Mat &result) {

	result.create(image.size(), image.type()); // allocate if necessary
	int nchannels= image.channels();

	for (int j= 1; j<image.rows-1; j++) { // for all rows (except first and last)

		const uchar* previous= image.ptr<const uchar>(j-1); // previous row
		const uchar* current= image.ptr<const uchar>(j);	// current row
		const uchar* next= image.ptr<const uchar>(j+1);		// next row

		uchar* output= result.ptr<uchar>(j);	// output row

		for (int i=nchannels; i<(image.cols-1)*nchannels; i++) {

			// apply sharpening operator
			*output++= cv::saturate_cast<uchar>(5*current[i]-current[i-nchannels]-current[i+nchannels]-previous[i]-next[i]); 
		}
	}

	// Set the unprocess pixels to 0
	result.row(0).setTo(cv::Scalar(0));
	result.row(result.rows-1).setTo(cv::Scalar(0));
	result.col(0).setTo(cv::Scalar(0));
	result.col(result.cols-1).setTo(cv::Scalar(0));
}

// same function but using iterator
// this one works only for gray-level image
void sharpenIterator(const cv::Mat &image, cv::Mat &result) {

	// must be a gray-level image
	CV_Assert(image.type() == CV_8UC1);

	// initialize iterators at row 1
	cv::Mat_<uchar>::const_iterator it= image.begin<uchar>()+image.cols;
	cv::Mat_<uchar>::const_iterator itend= image.end<uchar>()-image.cols;
	cv::Mat_<uchar>::const_iterator itup= image.begin<uchar>();
	cv::Mat_<uchar>::const_iterator itdown= image.begin<uchar>()+2*image.cols;

	// setup output image and iterator
	result.create(image.size(), image.type()); // allocate if necessary
	cv::Mat_<uchar>::iterator itout= result.begin<uchar>()+result.cols;

	for ( ; it!= itend; ++it, ++itout, ++itup, ++itdown) {

			*itout= cv::saturate_cast<uchar>(*it *5 - *(it-1)- *(it+1)- *itup - *itdown); 
	}

	// Set the unprocessed pixels to 0
	result.row(0).setTo(cv::Scalar(0));
	result.row(result.rows-1).setTo(cv::Scalar(0));
	result.col(0).setTo(cv::Scalar(0));
	result.col(result.cols-1).setTo(cv::Scalar(0));
}

// using kernel
void sharpen2D(const cv::Mat &image, cv::Mat &result) {

	// Construct kernel (all entries initialized to 0)
	cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
	// assigns kernel values
	kernel.at<float>(1,1)= 5.0;
	kernel.at<float>(0,1)= -1.0;
	kernel.at<float>(2,1)= -1.0;
	kernel.at<float>(1,0)= -1.0;
	kernel.at<float>(1,2)= -1.0;

	//filter the image
	cv::filter2D(image,result,image.depth(),kernel);
}

int main()
{
	// test sharpen function

	cv::Mat image= cv::imread("boldt.jpg");
	if (!image.data)
		return 0; 

	cv::Mat result;

	double time= static_cast<double>(cv::getTickCount());
	sharpen(image, result);
	time= (static_cast<double>(cv::getTickCount())-time)/cv::getTickFrequency();
	std::cout << "time= " << time << std::endl;

	cv::namedWindow("Image");
	cv::imshow("Image",result);

	// test sharpenIterator

    // open the image in gray-level
	image= cv::imread("boldt.jpg",0);

	time = static_cast<double>(cv::getTickCount());
	sharpenIterator(image, result);
	time= (static_cast<double>(cv::getTickCount())-time)/cv::getTickFrequency();
	std::cout << "time 3= " << time << std::endl;

	cv::namedWindow("Sharpened Image");
	cv::imshow("Sharpened Image",result);

	// test sharpen2D

	image= cv::imread("boldt.jpg");

	time = static_cast<double>(cv::getTickCount());
	sharpen2D(image, result);
	time= (static_cast<double>(cv::getTickCount())-time)/cv::getTickFrequency();
	std::cout << "time 2D= " << time << std::endl;

	cv::namedWindow("Image 2D");
	cv::imshow("Image 2D",result);

	cv::waitKey();

	return 0;
}


