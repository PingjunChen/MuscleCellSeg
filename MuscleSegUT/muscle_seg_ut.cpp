/********************************************************************
    Copyright:  BICI2
    Created:    **:**:**** **:**
    Filename:   muscle_seg_ut.cpp
    Author:     Pingjun Chen

    Purpose:    muscle segmentation unittest
*********************************************************************/

#include "gtest/gtest.h"
#include "muscle_seg.h"


//TEST(TESTMuscleSeg, TestMuscleSeg)
//{
//    clock_t begin, end;
//    
//    std::string edge_model_path = "..\\..\\Data\\Model\\EdgeModel\\";;
//    std::string opts_path = edge_model_path + "edgemodel_opts.txt";
//    std::string thrs_path = edge_model_path + "edgemodel_thrs.bin";
//    std::string fids_path = edge_model_path + "edgemodel_fids.bin";
//    std::string child_path = edge_model_path + "edgemodel_child.bin";
//    std::string count_path = edge_model_path + "edgemodel_count.bin";
//    std::string depth_path = edge_model_path + "edgemodel_depth.bin";
//    std::string ebins_path = edge_model_path + "edgemodel_eBins.bin";
//    std::string ebnds_path = edge_model_path + "edgemodel_eBnds.bin";
//
//    std::string seg_model_path = "..\\..\\Data\\Model\\SegModel\\";
//    seg_model_path += "seg_model.txt";
//
//
//    std::string d_data = "..\\..\\Data\\";
//    std::string img_path = d_data + "069.bmp";
//
//    cv::Mat test_img = cv::imread(img_path);
//    // Step 1: Random Forest Edge Detection
//    bici2::EdgeDetect ed;
//    std::cout << "Start Loading Edge Models ... \n";
//    begin = clock();
//    ed.SetModel(opts_path, thrs_path, fids_path, child_path, count_path,
//        depth_path, ebins_path, ebnds_path);
//    end = clock();
//    std::cout << "Loading Edge Models Finished, it took " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
//
//    // edge detection
//    cv::Mat rf_map = ed.GetRandomForestResult(test_img);
//
//    // Step 2: Frangi, UCM and Hierachical Seg
//    std::cout << "Start Loading Seg Models ... \n";
//    begin = clock();
//    bici2::CellModel cell_info;
//    cell_info.min_area = 10;
//    cell_info.max_area = std::numeric_limits<int>::max();
//    cell_info.min_solidity = 0.8;
//    bici2::RFmodel model_parse;
//    model_parse.ParseModelFromTxt(seg_model_path);
//    cell_info.cell_model = model_parse.rf_model;
//    end = clock();
//    std::cout << "Loading Seg Models Finished, it took " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
//
//    // MuscleSeg instance initilization
//    bici2::MuscleSeg seg1;
//    // Apply segmentation
//    seg1.ApplyHierarchicalSegmentation(cell_info, test_img, rf_map);
//
//    // Show the final result
//    cv::Mat imgWithContour = seg1.DrawContoursWithScore(test_img);
//    std::string saving_path = img_path.substr(0, img_path.rfind(".")) + "_contour.png";
//    cv::imwrite(saving_path, imgWithContour);
//
//    seg1.SaveContourToLocal(img_path);
//    
//    int a = 1;
//}

TEST(TESTBatchProcess, TestMuscleSeg)
{
    //std::string d_data = "..\\..\\..\\Data\\MuscleDataset\\";

    std::cout << "Start Loading Edge Models ... \n";
    std::string edge_model_path = "..\\..\\Data\\Model\\EdgeModel\\";
    std::string opts_path = edge_model_path + "edgemodel_opts.txt";
    std::string thrs_path = edge_model_path + "edgemodel_thrs.bin";
    std::string fids_path = edge_model_path + "edgemodel_fids.bin";
    std::string child_path = edge_model_path + "edgemodel_child.bin";
    std::string count_path = edge_model_path + "edgemodel_count.bin";
    std::string depth_path = edge_model_path + "edgemodel_depth.bin";
    std::string ebins_path = edge_model_path + "edgemodel_eBins.bin";
    std::string ebnds_path = edge_model_path + "edgemodel_eBnds.bin";
    bici2::EdgeDetect ed;
    ed.SetModel(opts_path, thrs_path, fids_path, child_path, count_path,
        depth_path, ebins_path, ebnds_path);
    std::cout << "Loading Edge Models Finish... \n";
    
    std::cout << "Start Loading Seg Models ... \n";
    std::string seg_model_path = "..\\..\\Data\\Model\\SegModel\\";
    seg_model_path += "seg_model.txt";
    bici2::CellModel cell_info;
    cell_info.min_area = 10;
    cell_info.max_area = std::numeric_limits<int>::max();
    cell_info.min_solidity = 0.8;
    bici2::RFmodel model_parse;
    model_parse.ParseModelFromTxt(seg_model_path);
    cell_info.cell_model = model_parse.rf_model;
    std::cout << "Loading Edge Models Finish... \n";
    // MuscleSeg instance initilization
    bici2::MuscleSeg seg1;


    std::string d_data = "..\\..\\Data\\Test10142016\\";
    std::string file_filter = "bmp";

    std::vector<std::string> fileList;
    bici2::GetDir(d_data, fileList, file_filter);

    std::cout << "Start Time: " << bici2::GetCurrentDateTime() << std::endl;

    std::vector<std::string> fileWithError;
    std::cout << "There are " << fileList.size() << " images in total." << std::endl;
    std::string img_path;
    clock_t begin, end;
    for (int ifile = 0; ifile < fileList.size(); ++ifile)
    {
        try
        {
            begin = clock();
            img_path = fileList[ifile];
            cv::Mat test_img = cv::imread(img_path);
            cv::Mat rf_map = ed.GetRandomForestResult(test_img);
            // Apply segmentation
            seg1.ApplyHierarchicalSegmentation(cell_info, test_img, rf_map);
            // Save the final result
            cv::Mat imgWithContour = seg1.DrawContoursWithScore(test_img);
            std::string saving_path = img_path.substr(0, img_path.rfind(".")) + "_contour.png";
            cv::imwrite(saving_path, imgWithContour);
            end = clock();
            std::cout << "Now Image " << ifile + 1 << "(" << fileList.size() << ")" 
                << " segmentation finished! Takes " << double(end - begin) / CLOCKS_PER_SEC << "s\n";
        }
        catch (std::exception& e)
        {
            fileWithError.push_back(fileList[ifile]);
            std::cout << e.what() << std::endl;
        }
    }


    if (fileWithError.size() != 0)
    {
        std::cout << "Program has problem with following images: " << std::endl;
        for (int ie = 0; ie < fileWithError.size(); ++ie)
        {
            std::cout << fileWithError[ie] << std::endl;
        }
    }
    else
    {
        std::cout << "Progarm run through all test cases: " << std::endl;
    }

    std::cout << "Finish Time: " << bici2::GetCurrentDateTime() << std::endl;

}