/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 11 of the book:
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
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "CameraCalibrator.h"

int main()
{
	cv::Mat image;
	std::vector<std::string> filelist;

	// generate list of chessboard image filename
	// named chessboard01 to chessboard27 in chessboard sub-dir
	for (int i=1; i<=27; i++) {

		std::stringstream str;
		str << "chessboards/chessboard" << std::setw(2) << std::setfill('0') << i << ".jpg";
		std::cout << str.str() << std::endl;

		filelist.push_back(str.str());
		image= cv::imread(str.str(),0);

		// cv::imshow("Board Image",image);	
		// cv::waitKey(100);
	}

	// Create calibrator object
    CameraCalibrator cameraCalibrator;
	// add the corners from the chessboard
	cv::Size boardSize(7,5);
	cameraCalibrator.addChessboardPoints(
		filelist,	// filenames of chessboard image
		boardSize, "Detected points");	// size of chessboard

	// calibrate the camera
    cameraCalibrator.setCalibrationFlag(true,true);
	cameraCalibrator.calibrate(image.size());

    // Exampple of Image Undistortion
    image = cv::imread(filelist[14],0);
	cv::Size newSize(static_cast<int>(image.cols*1.5), static_cast<int>(image.rows*1.5));
	cv::Mat uImage= cameraCalibrator.remap(image, newSize);

	// display camera matrix
	cv::Mat cameraMatrix= cameraCalibrator.getCameraMatrix();
	std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
	std::cout << cameraMatrix.at<double>(0,0) << " " << cameraMatrix.at<double>(0,1) << " " << cameraMatrix.at<double>(0,2) << std::endl;
	std::cout << cameraMatrix.at<double>(1,0) << " " << cameraMatrix.at<double>(1,1) << " " << cameraMatrix.at<double>(1,2) << std::endl;
	std::cout << cameraMatrix.at<double>(2,0) << " " << cameraMatrix.at<double>(2,1) << " " << cameraMatrix.at<double>(2,2) << std::endl;

	cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);
	cv::namedWindow("Undistorted Image");
    cv::imshow("Undistorted Image", uImage);

	// Store everything in a xml file
	cv::FileStorage fs("calib.xml", cv::FileStorage::WRITE);
	fs << "Intrinsic" << cameraMatrix;
	fs << "Distortion" << cameraCalibrator.getDistCoeffs();

	cv::waitKey();
	return 0;
}
