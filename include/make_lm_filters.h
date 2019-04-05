/******************************************************************************
Copyright:  BICI2
Created:    21:4:2016 0:26
Filename:   MakeLMFilters.h
Author:     Pingjun Chen

Purpose:    MakeLMFilters definition
******************************************************************************/

#ifndef MUSCLEMINER_MAKELMFILTERS_H_
#define MUSCLEMINER_MAKELMFILTERS_H_

#include <cmath>

#include <vector>

#include "opencv2/opencv.hpp"

#include "export.h"

namespace bici2
{
    class MUSCLEMINER_EXPORT MakeLMFilters
    {
    public:
        explicit MakeLMFilters(int num_orient = 8, int max_support = 49);
        ~MakeLMFilters();
        cv::Mat GenerateLMfilters();  // create LML filter banks
    
    private:
        cv::Mat MakeFilter(float scale, int xphase, int yphase,
            const std::vector<cv::Point2f>& pts, int sup);
        std::vector<float> Gauss1d(float sigma, float mean,
            const std::vector<float>& seqs, int ord);
        void Normalize(const cv::Mat& filter_in, cv::Mat& filter_out);
        cv::Mat FspecialLoG(int win_size, float sigma);

    private:
        int num_orient_;
        int max_support_;
    };
}

#endif // MUSCLEMINER_MAKELMFILTERS_H_