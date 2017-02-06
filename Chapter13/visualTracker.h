/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 13 of the book:
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

#if !defined FTRACKER
#define FTRACKER

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/tracking/tracker.hpp>

#include "videoprocessor.h"

class VisualTracker : public FrameProcessor {
	
	cv::Ptr<cv::Tracker> tracker;
	cv::Rect2d box;
	bool reset;

  public:

	// constructor specifying the tracker to be used
	VisualTracker(cv::Ptr<cv::Tracker> tracker) : 
		             reset(true), tracker(tracker) {}

	// set the bounding box to initiate tracking
	void setBoundingBox(const cv::Rect2d& bb) {

		box = bb;
		reset = true;
	}
	
	// callback processing method
	void process(cv:: Mat &frame, cv:: Mat &output) {

		if (reset) { // new tracking session
			reset = false;

			tracker->init(frame, box);


		} else { // update the target's position
		
			tracker->update(frame, box);
		}

		// draw bounding box on current frame
		frame.copyTo(output);
		cv::rectangle(output, box, cv::Scalar(255, 255, 255), 2);
	}
};

#endif
