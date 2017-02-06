/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 10 of the book:
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

#if !defined TMATCHER
#define TMATCHER

// set to 1 to view match results, 0 otherwise
#define VERBOSE 1

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

class TargetMatcher {

  private:

	  // pointer to the feature point detector object
	  cv::Ptr<cv::FeatureDetector> detector;
	  // pointer to the feature descriptor extractor object
	  cv::Ptr<cv::DescriptorExtractor> descriptor;
	  cv::Mat target;      // target image
	  int normType;        // to compare descriptor vectors
	  double distance;     // min reprojection error
	  int numberOfLevels;  // pyramid size
	  double scaleFactor;  // scale between levels
	  // the pyramid of target images and its keypoints
	  std::vector<cv::Mat> pyramid; 
	  std::vector<std::vector<cv::KeyPoint>> pyrKeypoints;
	  std::vector<cv::Mat> pyrDescriptors;

	  // create a pyramid of target images
	  void createPyramid() {

		  // create the pyramid of target images
		  pyramid.clear();
		  cv::Mat layer(target);
		  for (int i = 0; i < numberOfLevels; i++) { // reduce size at each layer
			  pyramid.push_back(target.clone());
			  resize(target, target, cv::Size(), scaleFactor, scaleFactor);
		  }

		  pyrKeypoints.clear();
		  pyrDescriptors.clear();
		  // keypoint detection and description in pyramid
		  for (int i = 0; i < numberOfLevels; i++) {
			  // detect target keypoints at level i
			  pyrKeypoints.push_back(std::vector<cv::KeyPoint>());
			  detector->detect(pyramid[i], pyrKeypoints[i]);
			  if (VERBOSE)
			     std::cout << "Interest points: target=" << pyrKeypoints[i].size() << std::endl;
			  // compute descriptor at level i
			  pyrDescriptors.push_back(cv::Mat());
			  descriptor->compute(pyramid[i], pyrKeypoints[i], pyrDescriptors[i]);
		  }
	  }

  public:

	  TargetMatcher(const cv::Ptr<cv::FeatureDetector> &detector,
 			        const cv::Ptr<cv::DescriptorExtractor> &descriptor = cv::Ptr<cv::DescriptorExtractor>(),
		            int numberOfLevels=8, double scaleFactor=0.9)
		  : detector(detector), descriptor(descriptor), normType(cv::NORM_L2), distance(1.0),
		    numberOfLevels(numberOfLevels), scaleFactor(scaleFactor) {

		  // in this case use the associated descriptor
		  if (!this->descriptor) {
			  this->descriptor = this->detector;
		  }
	  }

	  // Set the norm to be used for matching
	  void setNormType(int norm) {

		  normType= norm;
	  }

	  // Set the minimum reprojection distance
	  void setReprojectionDistance(double d) {

		  distance= d;
	  }

	  // Set the target image
	  void setTarget(const cv::Mat t) {

		  if (VERBOSE)
		     cv::imshow("Target", t);
		  target= t;
		  createPyramid();
	  }

	  // Identify good matches using RANSAC
	  // Return homography matrix and output matches
	  cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		                 std::vector<cv::KeyPoint>& keypoints1, 
						 std::vector<cv::KeyPoint>& keypoints2,
					     std::vector<cv::DMatch>& outMatches) {

		// Convert keypoints into Point2f	
		std::vector<cv::Point2f> points1, points2;	
		outMatches.clear();
		for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
			 it!= matches.end(); ++it) {

			 // Get the position of left keypoints
			 points1.push_back(keypoints1[it->queryIdx].pt);
			 // Get the position of right keypoints
			 points2.push_back(keypoints2[it->trainIdx].pt);
	    }

		// Find the homography between image 1 and image 2
		std::vector<uchar> inliers(points1.size(),0);
		cv::Mat homography= cv::findHomography(
			points1,points2, // corresponding points
		    inliers,         // match status (inlier or outlier)  
			cv::RHO,	     // RHO method
		    distance);       // max distance to reprojection point
	
		// extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator itIn= inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM= matches.begin();
		// for all matches
		for ( ;itIn!= inliers.end(); ++itIn, ++itM) {

			if (*itIn) { // it is a valid match

				outMatches.push_back(*itM);
			}
		}

		return homography;
	  }

	  // detect the defined planar target in an image
	  // returns the homography and
	  // the 4 corners of the detected target
	  cv::Mat detectTarget(const cv::Mat& image, 
		  // position of the target corners (clock-wise)
		  std::vector<cv::Point2f>& detectedCorners) {

		  // 1. detect image keypoints
		  std::vector<cv::KeyPoint> keypoints;
		  detector->detect(image, keypoints);
		  if (VERBOSE)
   		     std::cout << "Interest points: image=" << keypoints.size() << std::endl;
		  // compute descriptors
		  cv::Mat descriptors;
		  descriptor->compute(image, keypoints, descriptors);
 	      std::vector<cv::DMatch> matches;

		  cv::Mat bestHomography;
		  cv::Size bestSize;
		  int maxInliers = 0;
		  cv::Mat homography;

		  // Construction of the matcher  
		  cv::BFMatcher matcher(normType);

		  // 2. robustly find homography for each pyramid level
		  for (int i = 0; i < numberOfLevels; i++) {
			  // find a RANSAC homography between target and image
			  matches.clear();

			  // match descriptors
			  matcher.match(pyrDescriptors[i], descriptors, matches);
			  if (VERBOSE)
				  std::cout << "Number of matches (level " << i << ")=" << matches.size() << std::endl;
			  // validate matches using RANSAC
			  std::vector<cv::DMatch> inliers;
			  homography = ransacTest(matches, pyrKeypoints[i], keypoints, inliers);
			  if (VERBOSE)
				  std::cout << "Number of inliers=" << inliers.size() << std::endl;

			  if (inliers.size() > maxInliers) { // we have a better H
				  maxInliers = inliers.size();
				  bestHomography = homography;
				  bestSize = pyramid[i].size();
			  }

			  if (VERBOSE) {
				  cv::Mat imageMatches;
				  cv::drawMatches(target, pyrKeypoints[i],  // 1st image and its keypoints
					  image, keypoints,  // 2nd image and its keypoints
					  inliers,			// the matches
					  imageMatches,		// the image produced
					  cv::Scalar(255, 255, 255),  // color of the lines
					  cv::Scalar(255, 255, 255),  // color of the keypoints
					  std::vector<char>(),
					  2);
				  cv::imshow("Target matches", imageMatches);
				  cv::waitKey();
			  }
		  }

		  // 3. find the corner position on the image using best homography
		  if (maxInliers > 8) { // the estimate is valid

			  // target corners at best size
			  std::vector<cv::Point2f> corners;
			  corners.push_back(cv::Point2f(0, 0));
			  corners.push_back(cv::Point2f(bestSize.width - 1, 0));
			  corners.push_back(cv::Point2f(bestSize.width - 1, bestSize.height - 1));
			  corners.push_back(cv::Point2f(0, bestSize.height - 1));

			  // reproject the target corners
			  cv::perspectiveTransform(corners, detectedCorners, bestHomography);
		  }

		  if (VERBOSE)
			  std::cout << "Best number of inliers=" << maxInliers << std::endl;
		  return bestHomography;
	  }	  
};

#endif
