/******************************************************************************
Copyright:  BICI2
Created:    10:5:2016 12:55
Filename:   muscle_seg.cpp
Author:     Pingjun Chen

Purpose:    Implementation of MusleSeg class
******************************************************************************/


#include "muscle_seg.h"

namespace bici2
{
    MuscleSeg::MuscleSeg(int n_layers, float scale_ratio, float score_thresh)
    {
        n_layers_ = n_layers;
        scale_ratio_ = scale_ratio;
        score_thresh_ = score_thresh;
    }

    MuscleSeg::~MuscleSeg()
    {}

    void MuscleSeg::SetEdgeMap(const cv::Mat& edge_map)
    {
        this->edge_map_ = edge_map.clone();
    }

    void MuscleSeg::SetUCMmat(const cv::Mat& ucm_mat)
    {
        this->ucm_mat_ = ucm_mat.clone();
    }


    cv::Mat MuscleSeg::GetEdgeMap()
    {
        return this->edge_map_;
    }

    // Get Frangi result
    void MuscleSeg::CalFrangiFilterResult(const cv::Mat& frangi_in)
    {
        Frangi frangi_filter;
        frangi_filter.SetMat(frangi_in);
        this->edge_map_ = frangi_filter.ApplyFrangi();
    }

    // Get UCM result
    void MuscleSeg::CalUCMmat()
    {
        UCM ucm_filter;
        ucm_filter.SetMat(this->edge_map_);
        this->ucm_mat_ = ucm_filter.ApplyUCM();
    }

    cv::Mat MuscleSeg::GetUCMmat()
    {
        return this->ucm_mat_;
    }
    
    void MuscleSeg::ApplyHierarchicalSegmentation(const CellModel& model,
        cv::Mat& img, cv::Mat& frangi_in)
    {
        clock_t begin, end;
        std::cout << "Start UCM ... \n";
        begin = clock();
        this->CalFrangiFilterResult(frangi_in);
        this->CalUCMmat();
        end = clock();

        cv::Mat tmp_edge_map = this->GetEdgeMap();
        cv::Mat tmp_ucm = this->GetUCMmat();

        std::cout << "UCM Finished ... \n";
        std::cout << "UCM takes " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
        
        std::cout << "Start Segmentation ... \n";
        begin = clock();
        DoSegmentation(model, img, this->ucm_mat_, this->edge_map_, this->n_layers_, 
                this->score_thresh_, this->contours_, this->node_scores_, this->scale_ratio_);
        end = clock();
        std::cout << "Segmentation Finished ... \n";
        std::cout << "Segmentation takes " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    }

    std::vector<std::vector<cv::Point>> MuscleSeg::GetContours()
    {
        return this->contours_;
    }

    std::vector<float> MuscleSeg::GetScores()
    {
        return this->node_scores_;
    }

    void MuscleSeg::SaveContourToLocal(std::string filepath)
    {
        std::string img_path_without_extension = filepath.substr(0, filepath.rfind("."));
        std::size_t found = img_path_without_extension.find_last_of("/\\");
        std::string img_name_without_extension;
        if (std::string::npos == found)
            img_name_without_extension = img_path_without_extension;
        else
            img_name_without_extension = img_path_without_extension.substr(found+1);

        std::vector<float> node_scores = GetScores();
        std::vector<std::vector<cv::Point>> contours = GetContours();
        assert(node_scores.size() == contours.size());

        std::string saving_path = img_path_without_extension + "_contours.txt";
        std::ofstream out(saving_path);
        // out << img_name_without_extension << std::endl;
        out << contours.size() << std::endl;
        // out << this->score_thresh_ << std::endl;
        out << 1.0 << std::endl;

        for (int ic = 0; ic < contours.size(); ++ic)
        {
            out << contours[ic].size() << std::endl;
            out << node_scores[ic] << std::endl;
            for (int ip = 0; ip < contours[ic].size(); ++ip)
            {
                out << contours[ic][ip].x << " " << contours[ic][ip].y << std::endl;
            }
        }

        out.close();
    }

    cv::Mat MuscleSeg::DrawContoursWithScore(const cv::Mat& img)
    {
        std::vector<std::vector<cv::Point>> contours = GetContours();
        std::vector<float> node_scores = GetScores();

        cv::Scalar ctr_colors = cv::Scalar(0, 255, 0);
        cv::Mat img_contours = img.clone();
        cv::drawContours(img_contours, contours, -1, ctr_colors, 3);
        cv::Scalar txt_colors = cv::Scalar(255, 50, 0);
        for (int i = 0; i < node_scores.size(); i++)
        {
            float sum_x = 0.0;
            float sum_y = 0.0;
            for (int j = 0; j < contours[i].size(); j++)
            {
                sum_x += contours[i][j].x;
                sum_y += contours[i][j].y;
            }
            float cent_x = sum_x / contours[i].size();
            float cent_y = sum_y / contours[i].size();
            cv::Point cent = cv::Point(round(cent_x), round(cent_y));
            std::string s_score = std::to_string(node_scores[i]);
            s_score.erase(s_score.end() - 4, s_score.end());
            cv::putText(img_contours, s_score, cent, 0, 0.4, txt_colors, 1);
        }

        //cv::namedWindow("Contours");
        //cv::imshow("Contours", img_contours);
        //
        //cv::waitKey(0);

        return img_contours;
    }

    void GetDir(std::string folder, std::vector<std::string> & files, std::string filter)
    {
        files.clear();
        FILE* pipe = NULL;
        std::string pCmd = "dir /B /S " + folder;
        char buf[256];

        if (NULL == (pipe = _popen(pCmd.c_str(), "rt")))
        {
            std::cout << "Cannot Open Folder" << std::endl;
            return;
        }

        while (!feof(pipe))
        {
            if (fgets(buf, 256, pipe) != NULL)
            {
                std::string curFileName = std::string(buf);

                if (!curFileName.empty() && curFileName[curFileName.length() - 1] == '\n') {
                    curFileName.erase(curFileName.length() - 1);
                }

                std::size_t pos = curFileName.find_last_of(".");
                if (curFileName.substr(pos + 1) == filter)
                {
                    files.push_back(curFileName);
                }
            }
        }
        _pclose(pipe);
    }

    std::string GetCurrentDateTime()
    {
        time_t     now = time(0);
        struct tm  tstruct;
        char       buf[80];
        tstruct = *localtime(&now);
        // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
        // for more information about date/time format
        strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

        return buf;
    }

}

void MuscleSeg(const char* img_path,
    const char* edge_model_path,
    const char* seg_model_path)
// void MuscleSeg()
{
    // edge model path
    std::string edge_model_dir = std::string(edge_model_path);
    // std::string edge_model_dir = ".\\Model\\EdgeModel\\";
    std::string opts_path = edge_model_dir + "edgemodel_opts.txt";
    std::string thrs_path = edge_model_dir + "edgemodel_thrs.bin";
    std::string fids_path = edge_model_dir + "edgemodel_fids.bin";
    std::string child_path = edge_model_dir + "edgemodel_child.bin";
    std::string count_path = edge_model_dir + "edgemodel_count.bin";
    std::string depth_path = edge_model_dir + "edgemodel_depth.bin";
    std::string ebins_path = edge_model_dir + "edgemodel_eBins.bin";
    std::string ebnds_path = edge_model_dir + "edgemodel_eBnds.bin";

    // seg model path
    std::string seg_model_dir = std::string(seg_model_path) + "seg_model.txt";
    //std::string seg_model_dir = "Model\\SegModel\\seg_model.txt";

    std::string img_path_str = std::string(img_path);
    // std::string img_path_str = "test.jpg";

    cv::Mat test_img = cv::imread(img_path_str);
    bici2::EdgeDetect ed;
    ed.SetModel(opts_path, thrs_path, fids_path, child_path, count_path,
        depth_path, ebins_path, ebnds_path);
    cv::Mat rf_map = ed.GetRandomForestResult(test_img);

    bici2::CellModel cell_info;
    cell_info.min_area = 10;
    cell_info.max_area = std::numeric_limits<int>::max();
    cell_info.min_solidity = 0.8;
    bici2::RFmodel model_parse;
    model_parse.ParseModelFromTxt(seg_model_dir);
    cell_info.cell_model = model_parse.rf_model;

    bici2::MuscleSeg seg1;
    seg1.ApplyHierarchicalSegmentation(cell_info, test_img, rf_map);
    //cv::Mat imgWithContour = seg1.DrawContoursWithScore(test_img);
    //// Show the final result
    //std::string saving_path = img_path_str.substr(0, img_path_str.rfind(".")) + "_contour.png";
    //cv::imwrite(saving_path, imgWithContour);
    seg1.SaveContourToLocal(img_path);
}