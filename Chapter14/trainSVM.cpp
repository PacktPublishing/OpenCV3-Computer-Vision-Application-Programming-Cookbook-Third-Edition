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

#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

// draw one HOG over one cell
void drawHOG(std::vector<float>::const_iterator hog, // iterator to the HOG
	         int numberOfBins,        // number of bins inHOG
	         cv::Mat &image,          // image of the cell
	         float scale=1.0) {       // lenght multiplier

	const float PI = 3.1415927;
	float binStep = PI / numberOfBins;
	float maxLength = image.rows;
	float cx = image.cols / 2.;
	float cy = image.rows / 2.;

	// for each bin
	for (int bin = 0; bin < numberOfBins; bin++) {

		// bin orientation
		float angle = bin*binStep;
		float dirX = cos(angle);
		float dirY = sin(angle);
		// length of line proportion to bin size
		float length = 0.5*maxLength* *(hog+bin);

		// drawing the line
		float x1 = cx - dirX * length * scale;
		float y1 = cy - dirY * length * scale;
		float x2 = cx + dirX * length * scale;
		float y2 = cy + dirY * length * scale;
		cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), CV_RGB(255, 255, 255), 1);
	}
}

// Draw HOG over an image
void drawHOGDescriptors(const cv::Mat &image,  // the input image 
	                    cv::Mat &hogImage,     // the resulting HOG image
	                    cv::Size cellSize,     // size of each cell (blocks are ignored)
	                    int nBins) {           // number of bins

	// block size is image size
	cv::HOGDescriptor hog(cv::Size((image.cols / cellSize.width) * cellSize.width, 
		                           (image.rows / cellSize.height) * cellSize.height),
		cv::Size((image.cols / cellSize.width) * cellSize.width,
			(image.rows / cellSize.height) * cellSize.height),	
		cellSize,    // block stride (ony 1 block here)
		cellSize,    // cell size
		nBins);      // number of bins

	// compute HOG
	std::vector<float> descriptors;
	hog.compute(image, descriptors);

	float scale= 2.0 / *std::max_element(descriptors.begin(), descriptors.end());

	hogImage.create(image.rows, image.cols, CV_8U);

	std::vector<float>::const_iterator itDesc= descriptors.begin();

	for (int i = 0; i < image.rows / cellSize.height; i++) {
		for (int j = 0; j < image.cols / cellSize.width; j++) {
			// draw wach cell
			hogImage(cv::Rect(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height));
			drawHOG(itDesc, nBins, hogImage(cv::Rect(j*cellSize.width, i*cellSize.height,
				                           cellSize.width, cellSize.height)), scale);
			itDesc += nBins;
		}
	}
}

int main()
{
	cv::Mat image = imread("girl.jpg", cv::IMREAD_GRAYSCALE);
	cv::imshow("Original image", image);

	cv::HOGDescriptor hog(cv::Size((image.cols / 16) * 16, (image.rows / 16) * 16), // size of the window
		cv::Size(16, 16),    // block size
		cv::Size(16, 16),    // block stride
		cv::Size(4, 4),      // cell size
		9);                  // number of bins

	std::vector<float> descriptors;

	// Draw a representation of HOG cells
	cv::Mat hogImage= image.clone();
	drawHOGDescriptors(image, hogImage, cv::Size(16, 16), 9);
	cv::imshow("HOG image", hogImage);

	// generate the filename
	std::vector<std::string> imgs;
	std::string prefix = "stopSamples/stop";
	std::string ext = ".png";

	// loading 8 positive samples
	std::vector<cv::Mat> positives;

	for (long i = 0; i < 8; i++) {

		std::string name(prefix);
		std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << i; name += ss.str();
		name += ext;

		positives.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
	}

	// the first 8 positive samples
	cv::Mat posSamples(2 * positives[0].rows, 4 * positives[0].cols, CV_8U);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++) {

			positives[i * 4 + j].copyTo(posSamples(cv::Rect(j*positives[i * 4 + j].cols, i*positives[i * 4 + j].rows, positives[i * 4 + j].cols, positives[i * 4 + j].rows)));

		}

	cv::imshow("Positive samples", posSamples);


	// loading 8 negative samples
	std::string nprefix = "stopSamples/neg";
	std::vector<cv::Mat> negatives;

	for (long i = 0; i < 8; i++) {

		std::string name(nprefix);
		std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << i; name += ss.str();
		name += ext;

		negatives.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
	}

	// the first 8 negative samples
	cv::Mat negSamples(2 * negatives[0].rows, 4 * negatives[0].cols, CV_8U);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++) {

			negatives[i * 4 + j].copyTo(negSamples(cv::Rect(j*negatives[i * 4 + j].cols, i*negatives[i * 4 + j].rows, negatives[i * 4 + j].cols, negatives[i * 4 + j].rows)));
		}

	cv::imshow("Negative samples", negSamples);

	// The HOG descriptor for stop sign detection
	cv::HOGDescriptor hogDesc(positives[0].size(), // size of the window
		cv::Size(8, 8),    // block size
		cv::Size(4, 4),    // block stride
		cv::Size(4, 4),    // cell size
		9);                // number of bins

	// compute first descriptor 
	std::vector<float> desc;
	hogDesc.compute(positives[0], desc);

	std::cout << "Positive sample size: " << positives[0].rows << "x" << positives[0].cols << std::endl;
	std::cout << "HOG descriptor size: " << desc.size() << std::endl;

	// the matrix of sample descriptors
	int featureSize = desc.size();
	int numberOfSamples = positives.size() + negatives.size();
	// create the matrix that will contain the samples HOG
	cv::Mat samples(numberOfSamples, featureSize, CV_32FC1);

	// fill first row with first descriptor
	for (int i = 0; i < featureSize; i++)
		samples.ptr<float>(0)[i] = desc[i];

	// compute descriptor of the positive samples
	for (int j = 1; j < positives.size(); j++) {
		hogDesc.compute(positives[j], desc);
		// fill the next row with current descriptor
		for (int i = 0; i < featureSize; i++)
			samples.ptr<float>(j)[i] = desc[i];
	}

	// compute descriptor of the negative samples
	for (int j = 0; j < negatives.size(); j++) {
		hogDesc.compute(negatives[j], desc);
		// fill the next row with current descriptor
		for (int i = 0; i < featureSize; i++)
			samples.ptr<float>(j + positives.size())[i] = desc[i];
	}

	// Create the labels
	cv::Mat labels(numberOfSamples, 1, CV_32SC1);
	// labels of positive samples
	labels.rowRange(0, positives.size()) = 1.0;   
	// labels of negative samples
	labels.rowRange(positives.size(), numberOfSamples) = -1.0; 

	// create SVM classifier
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);

	// prepare the training data
	cv::Ptr<cv::ml::TrainData> trainingData =
		cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE, labels);

	// SVM training
	svm->train(trainingData);

	cv::Mat queries(4, featureSize, CV_32FC1);

	// fill the rows with query descriptors
	hogDesc.compute(cv::imread("stopSamples/stop08.png", cv::IMREAD_GRAYSCALE), desc);
	for (int i = 0; i < featureSize; i++)
		queries.ptr<float>(0)[i] = desc[i];
	hogDesc.compute(cv::imread("stopSamples/stop09.png", cv::IMREAD_GRAYSCALE), desc);
	for (int i = 0; i < featureSize; i++)
		queries.ptr<float>(1)[i] = desc[i];
	hogDesc.compute(cv::imread("stopSamples/neg08.png", cv::IMREAD_GRAYSCALE), desc);
	for (int i = 0; i < featureSize; i++)
		queries.ptr<float>(2)[i] = desc[i];
	hogDesc.compute(cv::imread("stopSamples/neg09.png", cv::IMREAD_GRAYSCALE), desc);
	for (int i = 0; i < featureSize; i++)
		queries.ptr<float>(3)[i] = desc[i];
	cv::Mat predictions;

	// Test the classifier with the last two pos and neg samples
	svm->predict(queries, predictions);

	for (int i = 0; i < 4; i++)
		std::cout << "query: " << i << ": " << ((predictions.at<float>(i) < 0.0)? "Negative" : "Positive") << std::endl;

	// People detection
	cv::Mat myImage = imread("person.jpg", cv::IMREAD_GRAYSCALE);

	// create the detector
	std::vector<cv::Rect> peoples;
	cv::HOGDescriptor peopleHog;
	peopleHog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	// detect peoples oin an image
	peopleHog.detectMultiScale(myImage, // input image
		peoples, // ouput list of bounding boxes 
		0,       // threshold to consider a detection to be positive 
		cv::Size(4, 4),   // window stride 
		cv::Size(32, 32), // image padding
		1.1,              // scale factor
		2);               // grouping threshold (0 means no grouping) 

	// draw detections on image
	std::cout << "Number of peoples detected: " << peoples.size() << std::endl;
	for (int i = 0; i < peoples.size(); i++)
		cv::rectangle(myImage, peoples[i], cv::Scalar(255, 255, 255), 2);

	cv::imshow("People detection", myImage);

	cv::waitKey();
}