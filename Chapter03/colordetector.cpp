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

#include "colordetector.h"
#include <vector>
	
cv::Mat ColorDetector::process(const cv::Mat &image) {

	  // re-allocate binary map if necessary
	  // same size as input image, but 1-channel
	  result.create(image.size(),CV_8U);

	  // Converting to Lab color space 
	  if (useLab)
		  cv::cvtColor(image, converted, CV_BGR2Lab);

	  // get the iterators
	  cv::Mat_<cv::Vec3b>::const_iterator it= image.begin<cv::Vec3b>();
	  cv::Mat_<cv::Vec3b>::const_iterator itend= image.end<cv::Vec3b>();
	  cv::Mat_<uchar>::iterator itout= result.begin<uchar>();

	  // get the iterators of the converted image 
	  if (useLab) {
		  it = converted.begin<cv::Vec3b>();
		  itend = converted.end<cv::Vec3b>();
	  }

	  // for each pixel
	  for ( ; it!= itend; ++it, ++itout) {
        
		// process each pixel ---------------------

		  // compute distance from target color
		  if (getDistanceToTargetColor(*it)<maxDist) {

			  *itout= 255;

		  } else {

			  *itout= 0;
		  }

        // end of pixel processing ----------------
	  }

	  return result;
}

