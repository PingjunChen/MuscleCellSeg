/******************************************************************************
Copyright:  BICI2
Created:    10:5:2016 11:50
Filename:   muscle_seg.h
Author:     Pingjun Chen

Purpose:    Integrate all splitting modules
******************************************************************************/

#ifndef MUSCLEMINER_MUSCLE_SEG_H_
#define MUSCLEMINER_MUSCLE_SEG_H_

#include <fstream>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "export.h"

#include "img_processing.h"
#include "edge_model.h"
#include "edge_detect.h"
#include "frangi.h"
#include "ucm.h"
#include "treemodel.h"
#include "hierarchicalimagesegmentation.h"


namespace bici2
{
    class MUSCLEMINER_EXPORT MuscleSeg
    {
    public:
        MuscleSeg(int n_layers = 5, float scale_ratio = 1.0, float score_thresh = 0.05);
        ~MuscleSeg();
        void SetEdgeMap(const cv::Mat& edge_map);
        void SetUCMmat(const cv::Mat& ucm_mat);

        //// Calcualte RandomForest Map
        //cv::Mat CalRandomForestMap(const cv::Mat& oriImg, std::string model_path);

        // Calculate Edge Map
        void CalFrangiFilterResult(const cv::Mat& frangi_in);
        cv::Mat GetEdgeMap();
        void CalUCMmat();
        cv::Mat GetUCMmat();
        void ApplyHierarchicalSegmentation(const CellModel& model,
            cv::Mat& img, cv::Mat& frangi_in);
        std::vector<std::vector<cv::Point>> GetContours();
        std::vector<float> GetScores();
        cv::Mat DrawContoursWithScore(const cv::Mat& img);
        void SaveContourToLocal(std::string filepath);

    private:
        cv::Mat edge_map_;
        cv::Mat ucm_mat_;

        int n_layers_;
        float scale_ratio_;
        float score_thresh_;

        std::vector<std::vector<cv::Point>> contours_;
        std::vector<float> node_scores_;
    };

    void MUSCLEMINER_EXPORT GetDir(std::string folder, std::vector<std::string> & files, std::string filter);
    std::string MUSCLEMINER_EXPORT GetCurrentDateTime();
}


extern "C" { void MUSCLEMINER_EXPORT MuscleSeg(const char* img_path, 
                                                const char* edge_model_path, 
                                                const char* seg_model_path);}

#endif // MUSCLEMINER_MUSCLE_SEG_H_


