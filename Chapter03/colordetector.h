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

#if !defined COLORDETECT
#define COLORDETECT

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ColorDetector {

  private:

	  // minimum acceptable distance
	  int maxDist; 

	  // target color
	  cv::Vec3b target; 

	  // image containing color converted image
	  cv::Mat converted;
	  bool useLab;

	  // image containing resulting binary map
	  cv::Mat result;

  public:

	  // empty constructor
	  // default parameter initialization here
	  ColorDetector() : maxDist(100), target(0,0,0), useLab(false) {}

	  // extra constructor for Lab color space example
	  ColorDetector(bool useLab) : maxDist(100), target(0,0,0), useLab(useLab) {}

	  // full constructor
	  ColorDetector(uchar blue, uchar green, uchar red, int mxDist=100, bool useLab=false): maxDist(mxDist), useLab(useLab) { 

		  // target color
		  setTargetColor(blue, green, red);
	  }

	  // Computes the distance from target color.
	  int getDistanceToTargetColor(const cv::Vec3b& color) const {
		  return getColorDistance(color, target);
	  }

	  // Computes the city-block distance between two colors.
	  int getColorDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) const {

		  return abs(color1[0]-color2[0])+
					abs(color1[1]-color2[1])+
					abs(color1[2]-color2[2]);

	 	  // Or:
		  // return static_cast<int>(cv::norm<int,3>(cv::Vec3i(color[0]-color2[0],color[1]-color2[1],color[2]-color2[2])));
		  
		  // Or:
		  // cv::Vec3b dist;
		  // cv::absdiff(color,color2,dist);
		  // return cv::sum(dist)[0];
	  }

	  // Processes the image. Returns a 1-channel binary image.
	  cv::Mat process(const cv::Mat &image);

	  cv::Mat operator()(const cv::Mat &image) {
	  
		  cv::Mat input;
		 
		  if (useLab) { // Lab conversion
			  cv::cvtColor(image, input, CV_BGR2Lab);
		  }
		  else {
			  input = image;
		  }

		  cv::Mat output;
		  // compute absolute difference with target color
		  cv::absdiff(input,cv::Scalar(target),output);
	      // split the channels into 3 images
	      std::vector<cv::Mat> images;
	      cv::split(output,images);
		  // add the 3 channels (saturation might occurs here)
	      output= images[0]+images[1]+images[2];
		  // apply threshold
          cv::threshold(output,  // input image
                      output,  // output image
                      maxDist, // threshold (must be < 256)
                      255,     // max value
                 cv::THRESH_BINARY_INV); // thresholding type
	
	      return output;
	  }

	  // Getters and setters

	  // Sets the color distance threshold.
	  // Threshold must be positive, otherwise distance threshold
	  // is set to 0.
	  void setColorDistanceThreshold(int distance) {

		  if (distance<0)
			  distance=0;
		  maxDist= distance;
	  }

	  // Gets the color distance threshold
	  int getColorDistanceThreshold() const {

		  return maxDist;
	  }

	  // Sets the color to be detected
	  // given in BGR color space
	  void setTargetColor(uchar blue, uchar green, uchar red) {

		  // BGR order
		  target = cv::Vec3b(blue, green, red);

		  if (useLab) {
			  // Temporary 1-pixel image
			  cv::Mat tmp(1, 1, CV_8UC3);
			  tmp.at<cv::Vec3b>(0, 0) = cv::Vec3b(blue, green, red);

			  // Converting the target to Lab color space 
			  cv::cvtColor(tmp, tmp, CV_BGR2Lab);

			  target = tmp.at<cv::Vec3b>(0, 0);
		  }
	  }

	  // Sets the color to be detected
	  void setTargetColor(cv::Vec3b color) {

		  target= color;
	  }

	  // Gets the color to be detected
	  cv::Vec3b getTargetColor() const {

		  return target;
	  }
};


#endif
