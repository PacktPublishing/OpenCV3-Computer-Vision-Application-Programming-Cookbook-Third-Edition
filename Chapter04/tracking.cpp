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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

#include "histogram.h"
#include "integral.h"

int main()
{
	// Open image
	cv::Mat image= cv::imread("bike55.bmp",0);
	// define image roi
	int xo=97, yo=112;
	int width=25, height=30;
	cv::Mat roi(image,cv::Rect(xo,yo,width,height));

	// compute sum
	// returns a Scalar to work with multi-channel images
	cv::Scalar sum= cv::sum(roi);
	std::cout << sum[0] << std::endl;

	// compute integral image
	cv::Mat integralImage;
	cv::integral(image,integralImage,CV_32S);
	// get sum over an area using three additions/subtractions
    int sumInt= integralImage.at<int>(yo+height,xo+width)
			      -integralImage.at<int>(yo+height,xo)
			      -integralImage.at<int>(yo,xo+width)
			      +integralImage.at<int>(yo,xo);
	std::cout << sumInt << std::endl;

	// histogram of 16 bins
	Histogram1D h;
	h.setNBins(16);
	// compute histogram over image roi 
	cv::Mat refHistogram= h.getHistogram(roi);

	cv::namedWindow("Reference Histogram");
	cv::imshow("Reference Histogram",h.getHistogramImage(roi,16));
	std::cout << refHistogram << std::endl;

	// first create 16-plane binary image
	cv::Mat planes;
	convertToBinaryPlanes(image,planes,16);
	// then compute integral image
	IntegralImage<float,16> intHisto(planes);


	// for testing compute a histogram of 16 bins with integral image
	cv::Vec<float,16> histogram= intHisto(xo,yo,width,height);
	std::cout<< histogram << std::endl;

	cv::namedWindow("Reference Histogram (2)");
	cv::Mat im= h.getImageOfHistogram(cv::Mat(histogram),16);
	cv::imshow("Reference Histogram (2)",im);	

	// search in second image
	cv::Mat secondImage= cv::imread("bike65.bmp",0);
	if (!secondImage.data)
		return 0; 

	// first create 16-plane binary image
	convertToBinaryPlanes(secondImage,planes,16);
	// then compute integral image
	IntegralImage<float,16> intHistogram(planes);

	// compute histogram of 16 bins with integral image (testing)
	histogram= intHistogram(135,114,width,height);
	std::cout<< histogram << std::endl;

	cv::namedWindow("Current Histogram");
	cv::Mat im2= h.getImageOfHistogram(cv::Mat(histogram),16);
	cv::imshow("Current Histogram",im2);	

	std::cout << "Distance= " << cv::compareHist(refHistogram,histogram, cv::HISTCMP_INTERSECT) << std::endl;

	double maxSimilarity=0.0;
	int xbest, ybest;
	// loop over a horizontal strip around girl location in initial image
	for (int y=110; y<120; y++) {
		for (int x=0; x<secondImage.cols-width; x++) {

	
			// compute histogram of 16 bins using integral image
			histogram= intHistogram(x,y,width,height);
			// compute distance with reference histogram
			double distance= cv::compareHist(refHistogram,histogram, cv::HISTCMP_INTERSECT);
			// find position of most similar histogram
			if (distance>maxSimilarity) {

				xbest= x;
				ybest= y;
				maxSimilarity= distance;
			}

			std::cout << "Distance(" << x << "," << y << ")=" << distance << std::endl;
		}
	}

    std::cout << "Best solution= (" << xbest << "," << ybest << ")=" << maxSimilarity << std::endl;

	// draw a rectangle around target object
	cv::rectangle(image,cv::Rect(xo,yo,width,height),0);
	cv::namedWindow("Initial Image");
	cv::imshow("Initial Image",image);

	cv::namedWindow("New Image");
	cv::imshow("New Image",secondImage);

	// draw rectangle at best location
	cv::rectangle(secondImage,cv::Rect(xbest,ybest,width,height),0);
	// draw rectangle around search area
	cv::rectangle(secondImage,cv::Rect(0,110,secondImage.cols,height+10),255);
	cv::namedWindow("Object location");
	cv::imshow("Object location",secondImage);
	
	cv::waitKey();
	
}