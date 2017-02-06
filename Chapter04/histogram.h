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

#if !defined HISTOGRAM
#define HISTOGRAM

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// To create histograms of gray-level images
class Histogram1D {

  private:

    int histSize[1];         // number of bins in histogram
	float hranges[2];        // range of values
    const float* ranges[1];  // pointer to the different value ranges
    int channels[1];         // channel number to be examined

  public:

	Histogram1D() {

		// Prepare default arguments for 1D histogram
		histSize[0]= 256;   // 256 bins
		hranges[0]= 0.0;    // from 0 (inclusive)
		hranges[1]= 256.0;  // to 256 (exclusive)
		ranges[0]= hranges; 
		channels[0]= 0;     // we look at channel 0
	}

	// Sets the channel on which histogram will be calculated.
	// By default it is channel 0.
	void setChannel(int c) {

		channels[0]= c;
	}

	// Gets the channel used.
	int getChannel() {

		return channels[0];
	}

	// Sets the range for the pixel values.
	// By default it is [0,256[
	void setRange(float minValue, float maxValue) {

		hranges[0]= minValue;
		hranges[1]= maxValue;
	}

	// Gets the min pixel value.
	float getMinValue() {

		return hranges[0];
	}

	// Gets the max pixel value.
	float getMaxValue() {

		return hranges[1];
	}

	// Sets the number of bins in histogram.
	// By default it is 256.
	void setNBins(int nbins) {

		histSize[0]= nbins;
	}

	// Gets the number of bins in histogram.
	int getNBins() {

		return histSize[0];
	}

	// Computes the 1D histogram.
	cv::Mat getHistogram(const cv::Mat &image) {

		cv::Mat hist;

		// Compute 1D histogram with calcHist
		cv::calcHist(&image, 
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			1,			// it is a 1D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}


    // Computes the 1D histogram and returns an image of it.
	cv::Mat getHistogramImage(const cv::Mat &image, int zoom = 1){

		// Compute histogram first
		cv::Mat hist = getHistogram(image);

		// Creates image
        return Histogram1D::getImageOfHistogram(hist, zoom);
	}

    // Stretches the source image using min number of count in bins.
    cv::Mat stretch(const cv::Mat &image, int minValue = 0) {

        // Compute histogram first
        cv::Mat hist = getHistogram(image);

        // find left extremity of the histogram
        int imin = 0;
        for (; imin < histSize[0]; imin++) {
            // ignore bins with less than minValue entries
            if (hist.at<float>(imin) > minValue)
                break;
        }

        // find right extremity of the histogram
        int imax = histSize[0] - 1;
        for (; imax >= 0; imax--) {

            // ignore bins with less than minValue entries
            if (hist.at<float>(imax) > minValue)
                break;
        }

        // Create lookup table
        int dims[1] = { 256 };
        cv::Mat lookup(1, dims, CV_8U);

        for (int i = 0; i<256; i++) {

            if (i < imin) lookup.at<uchar>(i) = 0;
            else if (i > imax) lookup.at<uchar>(i) = 255;
            else lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin));
        }

        // Apply lookup table
        cv::Mat result;
        result = applyLookUp(image, lookup);

        return result;
    }

    // Stretches the source image using percentile.
    cv::Mat stretch(const cv::Mat &image, float percentile) {

        // number of pixels in percentile
        float number= image.total()*percentile;

        // Compute histogram first
        cv::Mat hist = getHistogram(image);

        // find left extremity of the histogram
        int imin = 0;
        for (float count=0.0; imin < 256; imin++) {
            // number of pixel at imin and below must be > number
            if ((count+=hist.at<float>(imin)) >= number)
                break;
        }

        // find right extremity of the histogram
        int imax = 255;
        for (float count=0.0; imax >= 0; imax--) {
            // number of pixel at imax and below must be > number
            if ((count += hist.at<float>(imax)) >= number)
                break;
        }

        // Create lookup table
        int dims[1] = { 256 };
        cv::Mat lookup(1, dims, CV_8U);

        for (int i = 0; i<256; i++) {

            if (i < imin) lookup.at<uchar>(i) = 0;
            else if (i > imax) lookup.at<uchar>(i) = 255;
            else lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin));
        }

        // Apply lookup table
        cv::Mat result;
        result = applyLookUp(image, lookup);

        return result;
    }

    // static methods

    // Create an image representing a histogram
    static cv::Mat getImageOfHistogram(const cv::Mat &hist, int zoom) {

        // Get min and max bin values
        double maxVal = 0;
        double minVal = 0;
        cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

        // get histogram size
        int histSize = hist.rows;

        // Square image on which to display histogram
        cv::Mat histImg(histSize*zoom, histSize*zoom, CV_8U, cv::Scalar(255));

        // set highest point at 90% of nbins (i.e. image height)
        int hpt = static_cast<int>(0.9*histSize);

        // Draw vertical line for each bin
        for (int h = 0; h < histSize; h++) {

            float binVal = hist.at<float>(h);
            if (binVal>0) {
                int intensity = static_cast<int>(binVal*hpt / maxVal);
                cv::line(histImg, cv::Point(h*zoom, histSize*zoom),
                    cv::Point(h*zoom, (histSize - intensity)*zoom), cv::Scalar(0), zoom);
            }
        }

        return histImg;
    }

    // Equalizes the source image.
    static cv::Mat equalize(const cv::Mat &image) {

		cv::Mat result;
		cv::equalizeHist(image,result);

		return result;
	}


	// Applies a lookup table transforming an input image into a 1-channel image
    static cv::Mat applyLookUp(const cv::Mat& image, // input image
      const cv::Mat& lookup) { // 1x256 uchar matrix

      // the output image
      cv::Mat result;

      // apply lookup table
      cv::LUT(image,lookup,result);

      return result;
    }

	// Applies a lookup table transforming an input image into a 1-channel image
	// this is a test version with iterator; always use function cv::LUT
    static cv::Mat applyLookUpWithIterator(const cv::Mat& image, const cv::Mat& lookup) {

		// Set output image (always 1-channel)
		cv::Mat result(image.rows,image.cols,CV_8U);
		cv::Mat_<uchar>::iterator itr= result.begin<uchar>();

		// Iterates over the input image
		cv::Mat_<uchar>::const_iterator it= image.begin<uchar>();
		cv::Mat_<uchar>::const_iterator itend= image.end<uchar>();

		// Applies lookup to each pixel
		for ( ; it!= itend; ++it, ++itr) {

			*itr= lookup.at<uchar>(*it);
		}

		return result;
	}
};


#endif
