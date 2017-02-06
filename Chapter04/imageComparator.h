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
#if !defined ICOMPARATOR
#define ICOMPARATOR

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "colorhistogram.h"

class ImageComparator {

  private:

	cv::Mat refH;       // reference histogram
	cv::Mat inputH;     // histogram of input image

	ColorHistogram hist; 
	int nBins; // number of bins used in each color channel

  public:

	ImageComparator() :nBins(8) {

	}

	// Set number of bins used when comparing the histograms
	void setNumberOfBins( int bins) {

		nBins= bins;
	}

	int getNumberOfBins() {

		return nBins;
	}

	// set and compute histogram of reference image
	void setReferenceImage(const cv::Mat& image) {

		hist.setSize(nBins);
		refH= hist.getHistogram(image);
	}

	// compare the image using their BGR histograms
	double compare(const cv::Mat& image) {

		inputH= hist.getHistogram(image);

		// histogram comparison using intersection
		return cv::compareHist(refH,inputH, cv::HISTCMP_INTERSECT);
	}
};


#endif
