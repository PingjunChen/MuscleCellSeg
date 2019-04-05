/******************************************************************************
Copyright:  BICI2
Created:    20:4:2016 18:42
Filename:   UCM.h
Author:     Pingjun Chen

Purpose:    UCM definition
******************************************************************************/


#ifndef MUSCLEMINER_UCM_H_
#define MUSCLEMINER_UCM_H_


#include <cmath>

#include <string>
#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "export.h"
#include "img_processing.h"
#include "make_lm_filters.h"

namespace bici2
{
    class MUSCLEMINER_EXPORT UCM
    {
    public:
        UCM();
        explicit UCM(std::string path);
        explicit UCM(const cv::Mat& ucm_input);
        void SetPath(std::string path);             // Set the path for the image
        void SetMat(const cv::Mat& ucm_input);      // Set input cv::Mat for UCM
        std::string GetPath() const;                // Get path of the image
        cv::Mat GetMat() const;                     // Get cv::Mat 
        ~UCM();
        void SaveToYML(std::string yml_path) const;       // Save cv::Mat to YML file

    public:
        void MaxMinMat();  // get the maximum and minimum value of Mat

    public:
        cv::Mat ApplyUCM();
        cv::Mat SuperContour4C(const cv::Mat& pb);
        cv::Mat CleanWaterShed(const cv::Mat& ws);
        void MexContourSides(const cv::Mat& nmax);
        
        // friend UCM operator+(UCM &os, int b);

    private:
        cv::Mat ApplyLMFilter();    // Applying LM filters to input image
        cv::Mat Contour2UCM(const cv::Mat& imgs_orient, std::string fmt="imageSize");
        cv::Mat CreateFinestPartition(const cv::Mat& imgs_orient);


    private:
        std::string imgpath_;
        cv::Mat img_;
    };
}

#endif // MUSCLEMINER_UCM_H_