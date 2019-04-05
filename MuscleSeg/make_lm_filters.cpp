/******************************************************************************
Copyright:  BICI2
Created:    21:4:2016 0:29
Filename:   make_lm_filters.cpp
Author:     Pingjun Chen

Purpose:    MakeLMFilters Implementation
******************************************************************************/


#include "make_lm_filters.h"

namespace bici2
{
    MakeLMFilters::MakeLMFilters(int num_orient /* = 8 */, int max_support /* = 49 */)
    {
        this->num_orient_ = num_orient;
        this->max_support_ = max_support;
    }

    MakeLMFilters::~MakeLMFilters()
    {}

    // return the LML filter banks of size 49*49*48 in F.
    cv::Mat MakeLMFilters::GenerateLMfilters()
    {
        // int max_sup = 49;  // Support of the largest filter (must be odd)
        int max_sup = this->max_support_;
        std::vector<float> sigmax(3); // Sigma_{x} for the oriented filters
        for (unsigned int ix = 0; ix < sigmax.size(); ++ix)
        {
            sigmax[ix] = static_cast<float>(pow(sqrt(2.0), (ix + 1)));
        }

        int num_orient = this->num_orient_;
        int num_rot_inv = 12;
        int num_bar = sigmax.size() * num_orient;
        int num_edge = sigmax.size() * num_orient;
        int num_filters = num_rot_inv + num_bar + num_edge;

        // initialize orgpts
        int half_sup = (max_sup - 1) / 2;
        std::vector<cv::Point2f> orgpts;
        for (int ix = -1 * half_sup; ix <= half_sup; ++ix)
        {
            for (int iy = half_sup; iy >= -1 * half_sup; --iy)
            {
                orgpts.push_back(cv::Point2f(ix, iy));
            }
        }

        // define filters
        // int filter_size[3] = { max_sup, max_sup, num_filters };
        // for 3d Mat, the order should be {depth, height, width} 
        int filter_size[3] = { num_filters, max_sup, max_sup };

        cv::Mat filters = cv::Mat(3, filter_size, CV_32F, cv::Scalar(0));
        cv::Mat filter = cv::Mat(max_sup, max_sup, CV_32F, cv::Scalar(0));
        int filter_count = 0;
        float angle = 0.0;
        float cos_angle = 0.0;
        float sin_angle = 0.0;
        for (int iscale = 0; iscale < sigmax.size(); ++iscale)
        {
            for (int iorient = 0; iorient < num_orient; ++iorient)
            {
                angle = CV_PI * iorient / num_orient;
                cos_angle = std::cos(angle);
                sin_angle = std::sin(angle);

                std::vector<cv::Point2f> rotpts;
                for (int ip = 0; ip < orgpts.size(); ++ip)
                {
                    // points rotation
                    rotpts.push_back(cv::Point2f(
                        cos_angle * orgpts[ip].x - sin_angle * orgpts[ip].y,
                        sin_angle * orgpts[ip].x + cos_angle * orgpts[ip].y));
                }
                // filter 1
                filter = MakeFilter(sigmax[iscale], 0, 1, rotpts, max_sup);
                std::memcpy(filters.data + filter_count*max_sup*max_sup*filters.elemSize(),
                    filter.data, max_sup*max_sup*filters.elemSize());
                // filter 2
                filter = MakeFilter(sigmax[iscale], 0, 2, rotpts, max_sup);
                std::memcpy(filters.data + (filter_count + num_edge)*max_sup*max_sup*filters.elemSize(),
                    filter.data, max_sup*max_sup*filters.elemSize());
                filter_count += 1;
            }
        }

        filter_count = num_bar + num_edge;
        sigmax.push_back(static_cast<float>(pow(sqrt(2.0), 4)));
        for (int iscale = 0; iscale < sigmax.size(); ++iscale)
        {
            // add gaussian filter
            filter = cv::getGaussianKernel(max_sup, sigmax[iscale], CV_32F);
            cv::mulTransposed(filter, filter, false);
            this->Normalize(filter, filter);
            std::memcpy(filters.data + filter_count*max_sup*max_sup*filters.elemSize(),
                filter.data, max_sup*max_sup*filters.elemSize());
            // add LoG filter
            filter = FspecialLoG(max_sup, sigmax[iscale]);
            this->Normalize(filter, filter);
            std::memcpy(filters.data + (filter_count + 1)*max_sup*max_sup*filters.elemSize(),
                filter.data, max_sup*max_sup*filters.elemSize());
            // add LoG filter
            filter = FspecialLoG(max_sup, sigmax[iscale] * 3.0);
            this->Normalize(filter, filter);
            std::memcpy(filters.data + (filter_count + 2)*max_sup*max_sup*filters.elemSize(),
                filter.data, max_sup*max_sup*filters.elemSize());

            filter_count += 3; // increase filter_count
        }

        return filters;
    }


    cv::Mat MakeLMFilters::MakeFilter(float scale, int xphase, int yphase,
        const std::vector<cv::Point2f>& pts, int sup)
    {
        std::vector<float> pts_x(pts.size());  // saving x coordinates
        std::vector<float> pts_y(pts.size());  // saving y coordinates
        for (int ip = 0; ip < pts.size(); ++ip)
        {
            pts_x[ip] = pts[ip].x;
            pts_y[ip] = pts[ip].y;
        }

        std::vector<float> kernel_x = this->Gauss1d(3.0 * scale, 0, pts_x, xphase);
        std::vector<float> kernel_y = this->Gauss1d(scale, 0, pts_y, yphase);

        std::vector<float> kernel_xy(kernel_y.size());
        for (int ip = 0; ip < kernel_xy.size(); ++ip)
        {
            kernel_xy[ip] = kernel_x[ip] * kernel_y[ip];
        }
        // cv::Mat filter(sup, sup, CV_32F, (void *)&kernel_xy[0]);
        cv::Mat filter(sup, sup, CV_32F, cv::Scalar(0.0));
        std::memcpy(filter.data, &kernel_xy[0], sup*sup*filter.elemSize());

        cv::transpose(filter, filter);
        this->Normalize(filter, filter);

        return filter;
    }

    std::vector<float> MakeLMFilters::Gauss1d(float sigma, float mean,
        const std::vector<float>& seqs, int ord)
    {
        std::vector<float> num(seqs.size());
        std::vector<float> kernel(seqs.size());

        float variance = std::pow(sigma, 2);
        float denom = 2 * variance;

        float tmp_seq = 0;
        for (int iseq = 0; iseq < seqs.size(); ++iseq)
        {
            tmp_seq = seqs[iseq] - mean;            // subtract mean
            num[iseq] = tmp_seq * tmp_seq;    // square
            kernel[iseq] = std::exp(-1.0*num[iseq] / denom) /
                std::pow(CV_PI*denom, 0.5);
            if (1 == ord)
            {
                kernel[iseq] *= -1.0 * (tmp_seq / variance);
            }
            else if (2 == ord)
            {
                kernel[iseq] *= (num[iseq] - variance) / (variance * variance);
            }
        }

        return kernel;
    }

    void MakeLMFilters::Normalize(const cv::Mat& filter_in, cv::Mat& filter_out)
    {
        filter_out = filter_in.clone();
        float mean_filter = cv::mean(filter_in)[0];
        filter_out -= mean_filter;
        float sum_filter = cv::sum(abs(filter_in))[0];
        filter_out /= sum_filter;
    }

    cv::Mat MakeLMFilters::FspecialLoG(int win_size, float sigma)
    {
        cv::Mat filter_x(win_size, win_size, CV_32F);
        for (int isize = 0; isize < win_size; ++isize)
        {
            for (int jsize = 0; jsize < win_size; ++jsize)
            {
                filter_x.at<float>(jsize, isize) =
                    (isize - (win_size - 1) / 2) * (isize - (win_size - 1) / 2);
            }
        }
        cv::Mat filter_y;
        cv::transpose(filter_x, filter_y);
        cv::Mat arg = -(filter_x + filter_y) / (2 * std::pow(sigma, 2));

        cv::Mat filter_log(win_size, win_size, CV_32F);
        for (int isize = 0; isize < win_size; ++isize)
        {
            for (int jsize = 0; jsize < win_size; ++jsize)
            {
                filter_log.at<float>(jsize, isize)
                    = std::pow(exp(1), (arg.at<float>(jsize, isize)));
            }
        }

        double min_val, max_val;
        cv::minMaxLoc(filter_log, &min_val, &max_val);
        cv::Mat tmp_mask = (filter_log > DBL_EPSILON * max_val) / 255;
        tmp_mask.convertTo(tmp_mask, filter_log.type());
        cv::multiply(tmp_mask, filter_log, filter_log);

        if (0 != cv::sum(filter_log)[0])
        {
            filter_log = filter_log / cv::sum(filter_log)[0];
        }

        cv::Mat tmp_filter = (filter_x + filter_y -
            2 * (std::pow(sigma, 2))) / (std::pow(sigma, 4));

        cv::multiply(filter_log, tmp_filter, tmp_filter);
        filter_log = tmp_filter - cv::sum(tmp_filter)[0] / (win_size*win_size);

        return filter_log;
    }
}