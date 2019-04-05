/******************************************************************************
Copyright:  BICI2
Created:    18:4:2016 11:18
Filename:   frangi.h
Author:     Shiv

Purpose:    Frangi filter
******************************************************************************/



#ifndef MUSCLEMINER_FRANGI_H_
#define MUSCLEMINER_FRANGI_H_

#include <cmath>

#include <string>
#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "export.h"

namespace bici2
{

	class MUSCLEMINER_EXPORT Frangi
	{
	public:
		Frangi();
		Frangi(std::string path);
		void SetPath(std::string path);
        void SetMat(const cv::Mat& img);
		cv::Mat GetMat() const;
		cv::Mat ApplyFrangi();

		~Frangi();

	private:
		void Hessian2D(cv::Mat& d_xx,  // move to private once tested
			cv::Mat& d_xy,
			cv::Mat& d_yy,
			const cv::Mat& img,
			const float sigma);
		void Eig2image(cv::Mat& lambda1, // move to private once tested
			cv::Mat& lambda2,
			cv::Mat& Ix,
			cv::Mat& Iy,
			const cv::Mat& d_xx,
			const cv::Mat& d_xy,
        const cv::Mat& d_yy);
        //  move to private once tested
        void FrangiFilter2D(cv::Mat& outIm, const cv::Mat& Im);  
        void RemoveSmallBlobs(cv::Mat& im, double size);
        cv::Mat Anisodiff2D(cv::Mat& img);

        std::string imgpath_;
        cv::Mat img_;
    };
}

#endif // MUSCLEMINER_FRANGI_H_
