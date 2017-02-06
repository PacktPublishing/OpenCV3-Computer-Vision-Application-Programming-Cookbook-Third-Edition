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

#if !defined MATCHER
#define MATCHER

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define NOCHECK      0
#define CROSSCHECK   1
#define RATIOCHECK   2
#define BOTHCHECK    3

class RobustMatcher {

  private:

	  // pointer to the feature point detector object
	  cv::Ptr<cv::FeatureDetector> detector;
	  // pointer to the feature descriptor extractor object
	  cv::Ptr<cv::DescriptorExtractor> descriptor;
	  int normType;
	  float ratio; // max ratio between 1st and 2nd NN
	  bool refineF; // if true will refine the F matrix
	  bool refineM; // if true will refine the matches (will refine F also)
	  double distance; // min distance to epipolar
	  double confidence; // confidence level (probability)

  public:

	  RobustMatcher(const cv::Ptr<cv::FeatureDetector> &detector, 
		            const cv::Ptr<cv::DescriptorExtractor> &descriptor= cv::Ptr<cv::DescriptorExtractor>())
		  : detector(detector), descriptor(descriptor),normType(cv::NORM_L2), 
		    ratio(0.8f), refineF(true), refineM(true), confidence(0.98), distance(1.0) {	  

		// in this case use the associated descriptor
		if (!this->descriptor) { 
			this->descriptor = this->detector;
		} 
	  }

	  // Set the feature detector
	  void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {

		  this->detector= detect;
	  }

	  // Set descriptor extractor
	  void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) {

		  this->descriptor= desc;
	  }

	  // Set the norm to be used for matching
	  void setNormType(int norm) {

		  normType= norm;
	  }

	  // Set the minimum distance to epipolar in RANSAC
	  void setMinDistanceToEpipolar(double d) {

		  distance= d;
	  }

	  // Set confidence level in RANSAC
	  void setConfidenceLevel(double c) {

		  confidence= c;
	  }

	  // Set the NN ratio
	  void setRatio(float r) {

		  ratio= r;
	  }

	  // if you want the F matrix to be recalculated
	  void refineFundamental(bool flag) {

		  refineF= flag;
	  }

	  // if you want the matches to be refined using F
	  void refineMatches(bool flag) {

		  refineM= flag;
	  }

	  // Clear matches for which NN ratio is > than threshold
	  // return the number of removed points 
	  // (corresponding entries being cleared, i.e. size will be 0)
      int ratioTest(const std::vector<std::vector<cv::DMatch> >& inputMatches,
		            std::vector<cv::DMatch>& outputMatches) {

		int removed=0;

        // for all matches
        for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator= inputMatches.begin();
			 matchIterator!= inputMatches.end(); ++matchIterator) {
				 
				 //   first best match/second best match
				 if ((matchIterator->size() > 1) && // if 2 NN has been identified 
					 (*matchIterator)[0].distance/(*matchIterator)[1].distance < ratio) {
			
					 // it is an acceptable match
					 outputMatches.push_back((*matchIterator)[0]);

				 } else {

					 removed++;
				 }
		}

		return removed;
	  }

	  // Insert symmetrical matches in symMatches vector
	  void symmetryTest(const std::vector<cv::DMatch>& matches1,
		                const std::vector<cv::DMatch>& matches2,
					    std::vector<cv::DMatch>& symMatches) {
			
		// for all matches image 1 -> image 2
		for (std::vector<cv::DMatch>::const_iterator matchIterator1= matches1.begin();
			 matchIterator1!= matches1.end(); ++matchIterator1) {

			// for all matches image 2 -> image 1
			for (std::vector<cv::DMatch>::const_iterator matchIterator2= matches2.begin();
				matchIterator2!= matches2.end(); ++matchIterator2) {

				// Match symmetry test
				if (matchIterator1->queryIdx == matchIterator2->trainIdx  && 
					matchIterator2->queryIdx == matchIterator1->trainIdx) {

						// add symmetrical match
						symMatches.push_back(*matchIterator1);
						break; // next match in image 1 -> image 2
				}
			}
		}
	  }

	  // Apply both ratio and symmetry test
	  // (often an over-kill)
      void ratioAndSymmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1,
                                const std::vector<std::vector<cv::DMatch> >& matches2,
					            std::vector<cv::DMatch>& outputMatches) {

		// Remove matches for which NN ratio is > than threshold

		// clean image 1 -> image 2 matches
		std::vector<cv::DMatch> ratioMatches1;
		int removed= ratioTest(matches1,ratioMatches1);
		std::cout << "Number of matched points 1->2 (ratio test) : " << ratioMatches1.size() << std::endl;
		// clean image 2 -> image 1 matches
		std::vector<cv::DMatch> ratioMatches2;
		removed= ratioTest(matches2,ratioMatches2);
		std::cout << "Number of matched points 1->2 (ratio test) : " << ratioMatches2.size() << std::endl;

		// Remove non-symmetrical matches
		symmetryTest(ratioMatches1,ratioMatches2,outputMatches);

		std::cout << "Number of matched points (symmetry test): " << outputMatches.size() << std::endl;
	  }

	  // Identify good matches using RANSAC
	  // Return fundamental matrix and output matches
	  cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		                 std::vector<cv::KeyPoint>& keypoints1, 
						 std::vector<cv::KeyPoint>& keypoints2,
					     std::vector<cv::DMatch>& outMatches) {

		// Convert keypoints into Point2f	
		std::vector<cv::Point2f> points1, points2;	

		for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
			 it!= matches.end(); ++it) {

			 // Get the position of left keypoints
			 points1.push_back(keypoints1[it->queryIdx].pt);
			 // Get the position of right keypoints
			 points2.push_back(keypoints2[it->trainIdx].pt);
	    }

		// Compute F matrix using RANSAC
		std::vector<uchar> inliers(points1.size(),0);
		cv::Mat fundamental= cv::findFundamentalMat(
			points1,points2, // matching points
		    inliers,         // match status (inlier or outlier)  
		    cv::FM_RANSAC,   // RANSAC method
		    distance,        // distance to epipolar line
		    confidence);     // confidence probability
	
		// extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator itIn= inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM= matches.begin();
		// for all matches
		for ( ;itIn!= inliers.end(); ++itIn, ++itM) {

			if (*itIn) { // it is a valid match

				outMatches.push_back(*itM);
			}
		}

		if (refineF || refineM) {
		// The F matrix will be recomputed with all accepted matches

			// Convert keypoints into Point2f for final F computation	
			points1.clear();
			points2.clear();
	
			for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();
				 it!= outMatches.end(); ++it) {

				 // Get the position of left keypoints
				 points1.push_back(keypoints1[it->queryIdx].pt);
				 // Get the position of right keypoints
				 points2.push_back(keypoints2[it->trainIdx].pt);
			}

			// Compute 8-point F from all accepted matches
			fundamental= cv::findFundamentalMat(
				points1,points2, // matching points
				cv::FM_8POINT); // 8-point method

			if (refineM) {

				std::vector<cv::Point2f> newPoints1, newPoints2;	
				// refine the matches
				correctMatches(fundamental,             // F matrix
					           points1, points2,        // original position
							   newPoints1, newPoints2); // new position
				for (int i=0; i< points1.size(); i++) {

					std::cout << "(" << keypoints1[outMatches[i].queryIdx].pt.x 
						      << "," << keypoints1[outMatches[i].queryIdx].pt.y 
							  << ") -> ";
					std::cout << "(" << newPoints1[i].x 
						      << "," << newPoints1[i].y << std::endl;
					std::cout << "(" << keypoints2[outMatches[i].trainIdx].pt.x 
						      << "," << keypoints2[outMatches[i].trainIdx].pt.y 
							  << ") -> ";
					std::cout << "(" << newPoints2[i].x 
						      << "," << newPoints2[i].y << std::endl;

					keypoints1[outMatches[i].queryIdx].pt.x= newPoints1[i].x;
					keypoints1[outMatches[i].queryIdx].pt.y= newPoints1[i].y;
					keypoints2[outMatches[i].trainIdx].pt.x= newPoints2[i].x;
					keypoints2[outMatches[i].trainIdx].pt.y= newPoints2[i].y;
				}
			}
		}


		return fundamental;
	  }

	  // Match feature points using RANSAC
	  // returns fundamental matrix and output match set
	  cv::Mat match(cv::Mat& image1, cv::Mat& image2, // input images 
		  std::vector<cv::DMatch>& matches, // output matches and keypoints
		  std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
		  int check=CROSSCHECK) {  // check type (symmetry or ratio or none or both)

		// 1. Detection of the feature points
		detector->detect(image1,keypoints1);
		detector->detect(image2,keypoints2);

		std::cout << "Number of feature points (1): " << keypoints1.size() << std::endl;
		std::cout << "Number of feature points (2): " << keypoints2.size() << std::endl;

		// 2. Extraction of the feature descriptors
		cv::Mat descriptors1, descriptors2;
		descriptor->compute(image1,keypoints1,descriptors1);
		descriptor->compute(image2,keypoints2,descriptors2);

		std::cout << "descriptor matrix size: " << descriptors1.rows << " by " << descriptors1.cols << std::endl;

		// 3. Match the two image descriptors
		//    (optionaly apply some checking method)
   
		// Construction of the matcher with crosscheck 
		cv::BFMatcher matcher(normType,            //distance measure
	                          check==CROSSCHECK);  // crosscheck flag
                             
		// vectors of matches
        std::vector<std::vector<cv::DMatch> > matches1;
        std::vector<std::vector<cv::DMatch> > matches2;
	    std::vector<cv::DMatch> outputMatches;

		// call knnMatch if ratio check is required
		if (check==RATIOCHECK || check==BOTHCHECK) {
			// from image 1 to image 2
			// based on k nearest neighbours (with k=2)
			matcher.knnMatch(descriptors1,descriptors2, 
				matches1, // vector of matches (up to 2 per entry) 
				2);		  // return 2 nearest neighbours

			std::cout << "Number of matched points 1->2: " << matches1.size() << std::endl;

			if (check==BOTHCHECK) {
				// from image 2 to image 1
				// based on k nearest neighbours (with k=2)
				matcher.knnMatch(descriptors2,descriptors1, 
					matches2, // vector of matches (up to 2 per entry) 
					2);		  // return 2 nearest neighbours

				std::cout << "Number of matched points 2->1: " << matches2.size() << std::endl;
			}

		} 
		
		// select check method
		switch (check) {

			case CROSSCHECK:
				matcher.match(descriptors1,descriptors2,outputMatches);
				std::cout << "Number of matched points 1->2 (after cross-check): " << outputMatches.size() << std::endl;
				break;
			case RATIOCHECK:
				ratioTest(matches1,outputMatches);
				std::cout << "Number of matched points 1->2 (after ratio test): " << outputMatches.size() << std::endl;
				break;
			case BOTHCHECK:
				ratioAndSymmetryTest(matches1,matches2,outputMatches);
				std::cout << "Number of matched points 1->2 (after ratio and cross-check): " << outputMatches.size() << std::endl;
				break;
			case NOCHECK:
			default:
				matcher.match(descriptors1,descriptors2,outputMatches);
				std::cout << "Number of matched points 1->2: " << outputMatches.size() << std::endl;
				break;
		}

		// 4. Validate matches using RANSAC
		cv::Mat fundamental= ransacTest(outputMatches, keypoints1, keypoints2, matches);
		std::cout << "Number of matched points (after RANSAC): " << matches.size() << std::endl;

		// return the found fundamental matrix
		return fundamental;
	}
	  
	 // Match feature points using RANSAC
	 // returns fundamental matrix and output match set
     // this is the simplified version presented in the book
	  cv::Mat matchBook(cv::Mat& image1, cv::Mat& image2, // input images 
		  std::vector<cv::DMatch>& matches, // output matches and keypoints
		  std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2) { 
			  
		// 1. Detection of the feature points
		detector->detect(image1,keypoints1);
		detector->detect(image2,keypoints2);

		// 2. Extraction of the feature descriptors
		cv::Mat descriptors1, descriptors2;
		descriptor->compute(image1,keypoints1,descriptors1);
		descriptor->compute(image2,keypoints2,descriptors2);

		// 3. Match the two image descriptors
		//    (optionnally apply some checking method)
   
		// Construction of the matcher with crosscheck 
		cv::BFMatcher matcher(normType,   //distance measure
	                          true);      // crosscheck flag
                             
		// match descriptors
	    std::vector<cv::DMatch> outputMatches;
		matcher.match(descriptors1,descriptors2,outputMatches);

		// 4. Validate matches using RANSAC
		cv::Mat fundamental= ransacTest(outputMatches, keypoints1, keypoints2, matches);

		// return the found fundemental matrix
		return fundamental;
	}

};

#endif
