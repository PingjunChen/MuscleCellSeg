/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   edge_detect.h
Author:     He Zhao

Purpose:    
******************************************************************************/

#ifndef MUSCLEMINER_EDGE_DETECT_H_
#define MUSCLEMINER_EDGE_DETECT_H_

#include <cmath>
#include <ctime>
#include <vector>
#include <string>

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

#include "export.h"
#include "edge_model.h"
#include "wrapper.h"

namespace bici2
{
	class MUSCLEMINER_EXPORT EdgeDetect
	{
	public:
		EdgeDetect();

		//init functions, loading img and model
		void SetImg(const cv::Mat& img);
        void SetModel(const std::string& opts_path,
            const std::string& thrs_path,
            const std::string& fids_path,
            const std::string& child_path,
            const std::string& count_path,
            const std::string& depth_path,
            const std::string& ebins_path,
            const std::string& ebnds_path);

		// main function
        cv::Mat GetRandomForestResult(const cv::Mat& ori_img);

        cv::Mat EdgeDetectfunction(const cv::Mat& img);

        //multiple calculate
        cv::Mat mutlipleCalculate(const cv::Mat& img);

		//imresample functoin type 1 and 2;
        cv::Mat ImResample(const cv::Mat& img, float scale);
        cv::Mat ImResample(const cv::Mat& img, std::vector<int> scale);
		
		//imPad FUNCITON
        cv::Mat imPad(const cv::Mat& I, std::vector<int> pad, std::string type);

		//edgeChns function
		//cv::Mat edgeChns(cv::Mat I, EdgeModelOpts opts);
        void edgeChns(const cv::Mat& I, EdgeModelOpts opts);
		//convTri function
        static cv::Mat convTri(const cv::Mat& img, int a);

		//edgeNms
		// cv::Mat EdgeNms(cv::Mat E, cv::Mat O, int r, int s);

		//rgb convert
        cv::Mat RbgConvert(const cv::Mat& inputImg, std::string colorSpace);

		// EdgeOrient function
		cv::Mat EdgeOrient(cv::Mat inputImg, int r);
		
		//GradientMag function
        cv::Mat GradientMag(const cv::Mat& img, int channels, int normRad, float normConst);
        cv::Mat GradientHist(const cv::Mat& M, const cv::Mat& O, int binSize, int nOrients, int softbin);
		
        ~EdgeDetect();

	private:
		 EdgeModel model_;
		 cv::Mat img_;
		 cv::Mat chnsReg;
		 cv::Mat chnsSim;
		 cv::Mat M;
		 cv::Mat O;
	};
}

#endif // MUSCLEMINER_EDGE_DETECT_H_