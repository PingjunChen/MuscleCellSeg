/******************************************************************************
Copyright:  BICI2
Created:    18:5:2016 11:18
Filename:   img_processing.h
Author:     Pingjun Chen

Purpose:    Saving common image processing functions
******************************************************************************/

#ifndef MUSCLEMINER_IMG_PROCESSING_H_
#define MUSCLEMINER_IMG_PROCESSING_H_

#include <cmath>
#include <ctime>
#include <list>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "water_shed.h"
#include "region_props.h"
#include "ucm_mean_pb.h"

#include "export.h"


namespace bici2
{
    // Display cv::Mat
    MUSCLEMINER_EXPORT void Display(const cv::Mat& img);
    // Load cv::Mat from YML file with specified variable
    MUSCLEMINER_EXPORT cv::Mat LoadFromYML(std::string yml_path, std::string var_name);
    MUSCLEMINER_EXPORT cv::Mat MorphClean(const cv::Mat& bin_img);
    // Find local minima of Mat
    MUSCLEMINER_EXPORT cv::Mat FindLocalMinima(const cv::Mat& image, int window_size);
    // Watershed function
    MUSCLEMINER_EXPORT cv::Mat WatershedFull(const cv::Mat& image, const cv::Mat& marker);
    //// Implementation of watershed according to matlab version
    //MUSCLEMINER_EXPORT cv::Mat ComputeWatershed(const cv::Mat& image, const cv::Mat& marker);
    // Find connected component 
    MUSCLEMINER_EXPORT cv::Mat CreateConnectedComponent(const cv::Mat& ws_wt2);
    // convert ucm_mean_pb mex function
    MUSCLEMINER_EXPORT cv::Mat UCMMeanPB(const cv::Mat& ws_wt2,
        const cv::Mat& labels);
    // normalize image with specfied paramters
    MUSCLEMINER_EXPORT cv::Mat NormalizeImg(const cv::Mat& img, std::string fmt="imageSize");
    // Calculate region properties
    void MUSCLEMINER_EXPORT RegionProps(const cv::Mat& in, std::vector<RegProps> &out,
        kRegProps kregProp1 = RP_BLANK_PROP, kRegProps kregProp2 = RP_BLANK_PROP,
        kRegProps kregProp3 = RP_BLANK_PROP, kRegProps kregProp4 = RP_BLANK_PROP);

}

#endif MUSCLEMINER_IMG_PROCESSING_H_