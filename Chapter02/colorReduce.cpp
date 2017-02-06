/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 2 of the book:
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// 1st version
// see recipe Scanning an image with pointers
void colorReduce(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line

      for (int j=0; j<nl; j++) {

          // get the address of row j
          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

            data[i]= data[i]/div*div + div/2;

            // end of pixel processing ----------------

          } // end of line
      }
}

// version with input/ouput images
// see recipe Scanning an image with pointers
void colorReduceIO(const cv::Mat &image, // input image
	               cv::Mat &result,      // output image
	               int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols; // number of columns
	int nchannels = image.channels(); // number of channels

	// allocate output image if necessary
	result.create(image.rows, image.cols, image.type());

	for (int j = 0; j<nl; j++) {

		// get the addresses of input and output row j
		const uchar* data_in = image.ptr<uchar>(j);
		uchar* data_out = result.ptr<uchar>(j);

		for (int i = 0; i<nc*nchannels; i++) {

			// process each pixel ---------------------

			data_out[i] = data_in[i] / div*div + div / 2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 1
// this version uses the dereference operator *
void colorReduce1(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line
	  uchar div2 = div >> 1; // div2 = div/2

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            
			  // process each pixel ---------------------

			  *data++= *data/div*div + div2;

			  // end of pixel processing ----------------

          } // end of line
      }
}

// Test 2
// this version uses the modulo operator
void colorReduce2(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line
	  uchar div2 = div >> 1; // div2 = div/2

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

                 int v= *data;
                 *data++= v - v%div + div2;

            // end of pixel processing ----------------

          } // end of line
      }
}

// Test 3
// this version uses a binary mask
void colorReduce3(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line
      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
      uchar div2= 1<<(n-1); // div2 = div/2

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

		  for (int i = 0; i < nc; i++) {

			  // process each pixel ---------------------

			  *data &= mask;     // masking
			  *data++ |= div2;   // add div/2

            // end of pixel processing ----------------

          } // end of line
      }
}


// Test 4
// this version uses direct pointer arithmetic with a binary mask
void colorReduce4(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line
      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      int step= image.step; // effective width
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
	  uchar div2 = div >> 1; // div2 = div/2

      // get the pointer to the image buffer
      uchar *data= image.data;

      for (int j=0; j<nl; j++) {

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

            *(data+i) &= mask;
            *(data+i) += div2;

            // end of pixel processing ----------------

          } // end of line

          data+= step;  // next line
      }
}

// Test 5
// this version recomputes row size each time
void colorReduce5(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<image.cols * image.channels(); i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div/2;

            // end of pixel processing ----------------

          } // end of line
      }
}

// Test 6
// this version optimizes the case of continuous image
void colorReduce6(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line

      if (image.isContinuous())  {
          // then no padded pixels
          nc= nc*nl;
          nl= 1;  // it is now a 1D array
       }

      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
	  uchar div2 = div >> 1; // div2 = div/2

     // this loop is executed only once
     // in case of continuous images
      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div2;

            // end of pixel processing ----------------

          } // end of line
      }
}

// Test 7
// this versions applies reshape on continuous image
void colorReduce7(cv::Mat image, int div=64) {

      if (image.isContinuous()) {
        // no padded pixels
        image.reshape(1,   // new number of channels
                      1) ; // new number of rows
      }
      // number of columns set accordingly

      int nl= image.rows; // number of lines
      int nc= image.cols*image.channels() ; // number of columns

      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
	  uchar div2 = div >> 1; // div2 = div/2

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div2;

            // end of pixel processing ----------------

          } // end of line
      }
}

// Test 8
// this version processes the 3 channels inside the loop with Mat_ iterators
void colorReduce8(cv::Mat image, int div=64) {

      // get iterators
      cv::Mat_<cv::Vec3b>::iterator it= image.begin<cv::Vec3b>();
      cv::Mat_<cv::Vec3b>::iterator itend= image.end<cv::Vec3b>();
	  uchar div2 = div >> 1; // div2 = div/2

      for ( ; it!= itend; ++it) {

        // process each pixel ---------------------

        (*it)[0]= (*it)[0]/div*div + div2;
        (*it)[1]= (*it)[1]/div*div + div2;
        (*it)[2]= (*it)[2]/div*div + div2;

        // end of pixel processing ----------------
      }
}

// Test 9
// this version uses iterators on Vec3b
void colorReduce9(cv::Mat image, int div=64) {

      // get iterators
      cv::MatIterator_<cv::Vec3b> it= image.begin<cv::Vec3b>();
      cv::MatIterator_<cv::Vec3b> itend= image.end<cv::Vec3b>();

      const cv::Vec3b offset(div/2,div/2,div/2);

      for ( ; it!= itend; ++it) {

        // process each pixel ---------------------

        *it= *it/div*div+offset;
        // end of pixel processing ----------------
      }
}

// Test 10
// this version uses iterators with a binary mask
void colorReduce10(cv::Mat image, int div=64) {

      // div must be a power of 2
      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
	  uchar div2 = div >> 1; // div2 = div/2

      // get iterators
      cv::Mat_<cv::Vec3b>::iterator it= image.begin<cv::Vec3b>();
      cv::Mat_<cv::Vec3b>::iterator itend= image.end<cv::Vec3b>();

      // scan all pixels
      for ( ; it!= itend; ++it) {

        // process each pixel ---------------------

        (*it)[0]&= mask;
        (*it)[0]+= div2;
        (*it)[1]&= mask;
        (*it)[1]+= div2;
        (*it)[2]&= mask;
        (*it)[2]+= div2;

        // end of pixel processing ----------------
      }
}

// Test 11
// this versions uses ierators from Mat_ 
void colorReduce11(cv::Mat image, int div=64) {

      // get iterators
      cv::Mat_<cv::Vec3b> cimage= image;
      cv::Mat_<cv::Vec3b>::iterator it=cimage.begin();
      cv::Mat_<cv::Vec3b>::iterator itend=cimage.end();
	  uchar div2 = div >> 1; // div2 = div/2

      for ( ; it!= itend; it++) {

        // process each pixel ---------------------

        (*it)[0]= (*it)[0]/div*div + div2;
        (*it)[1]= (*it)[1]/div*div + div2;
        (*it)[2]= (*it)[2]/div*div + div2;

        // end of pixel processing ----------------
      }
}


// Test 12
// this version uses the at method
void colorReduce12(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols; // number of columns
	  uchar div2 = div >> 1; // div2 = div/2

      for (int j=0; j<nl; j++) {
          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

                  image.at<cv::Vec3b>(j,i)[0]=	 image.at<cv::Vec3b>(j,i)[0]/div*div + div2;
                  image.at<cv::Vec3b>(j,i)[1]=	 image.at<cv::Vec3b>(j,i)[1]/div*div + div2;
                  image.at<cv::Vec3b>(j,i)[2]=	 image.at<cv::Vec3b>(j,i)[2]/div*div + div2;

            // end of pixel processing ----------------

          } // end of line
      }
}


// Test 13
// this version uses Mat overloaded operators
void colorReduce13(cv::Mat image, int div=64) {

      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      // perform color reduction
      image=(image&cv::Scalar(mask,mask,mask))+cv::Scalar(div/2,div/2,div/2);
}

// Test 14
// this version uses a look up table
void colorReduce14(cv::Mat image, int div=64) {

      cv::Mat lookup(1,256,CV_8U);

      for (int i=0; i<256; i++) {

        lookup.at<uchar>(i)= i/div*div + div/2;
      }

      cv::LUT(image,lookup,image);
}

#define NTESTS 15
#define NITERATIONS 10

int main()
{
	// read the image
	cv::Mat image = cv::imread("boldt.jpg");

	// time and process the image
	const int64 start = cv::getTickCount();
	colorReduce(image, 64);
	//Elapsed time in seconds
	double duration = (cv::getTickCount() - start) / cv::getTickFrequency();

	// display the image
	std::cout << "Duration= " << duration << "secs" << std::endl;
	cv::namedWindow("Image");
	cv::imshow("Image", image);

	cv::waitKey();

	// test different versions of the function

	int64 t[NTESTS], tinit;
	// timer values set to 0
	for (int i = 0; i<NTESTS; i++)
		t[i] = 0;

	cv::Mat images[NTESTS];
	cv::Mat result;

	// the versions to be tested
	typedef void(*FunctionPointer)(cv::Mat, int);
	FunctionPointer functions[NTESTS] = { colorReduce, colorReduce1, colorReduce2, colorReduce3, colorReduce4,
										  colorReduce5, colorReduce6, colorReduce7, colorReduce8, colorReduce9,
										  colorReduce10, colorReduce11, colorReduce12, colorReduce13, colorReduce14};
	// repeat the tests several times
	int n = NITERATIONS;
	for (int k = 0; k<n; k++) {

		std::cout << k << " of " << n << std::endl;

		// test each version
		for (int c = 0; c < NTESTS; c++) {

			images[c] = cv::imread("boldt.jpg");

			// set timer and call function
			tinit = cv::getTickCount();
			functions[c](images[c], 64);
			t[c] += cv::getTickCount() - tinit;

			std::cout << ".";
		}

		std::cout << std::endl;
	}

	// short description of each function
	std::string descriptions[NTESTS] = {
		"original version:",
		"with dereference operator:",
		"using modulo operator:",
		"using a binary mask:",
		"direct ptr arithmetic:",
		"row size recomputation:",
		"continuous image:",
		"reshape continuous image:",
		"with iterators:",
		"Vec3b iterators:",
		"iterators and mask:",
		"iterators from Mat_:",
		"at method:",
		"overloaded operators:",
		"look-up table:",
	};

	for (int i = 0; i < NTESTS; i++) {

		cv::namedWindow(descriptions[i]);
		cv::imshow(descriptions[i], images[i]);
	}

	// print average execution time
	std::cout << std::endl << "-------------------------------------------" << std::endl << std::endl;
	for (int i = 0; i < NTESTS; i++) {

		std::cout << i << ". " << descriptions[i] << 1000.*t[i] / cv::getTickFrequency() / n << "ms" << std::endl;
	}

	cv::waitKey();
	return 0;
}