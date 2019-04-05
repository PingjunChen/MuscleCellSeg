/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   hierarchicalimagesegmentation.h
Author:     Mason McGough

Purpose:
******************************************************************************/

#ifndef MUSCLEMINER_HIERARCHICALIMAGESEGMENTATION_H_
#define MUSCLEMINER_HIERARCHICALIMAGESEGMENTATION_H_

#include <time.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "export.h"
#include "treemodel.h"
#include "opencvsparsematrixmath.h"
#include "feature_extraction.h"

namespace bici2 
{
	const int SORT_ASCEND = 0;
	const int SORT_DESCEND = 1;

	void MUSCLEMINER_EXPORT ExtractVectorInt(const std::vector<int>& original_vector, const std::vector<int>& indices, std::vector<int>& output_vector, const int offset = 0);
	void MUSCLEMINER_EXPORT ExtractVectorFloat(const std::vector<float>& original_vector, const std::vector<int>& indices, std::vector<float>& output_vector, const int offset = 0);
	void MUSCLEMINER_EXPORT ExtractVectorContour(const std::vector<std::vector<cv::Point>>& original_vector, const std::vector<int>& indices, std::vector<std::vector<cv::Point>>& output_vector, const int offset = 0);
	void MUSCLEMINER_EXPORT SortIndices(const std::vector<float>& original_vector, std::vector<float>& output_vector, std::vector<int>& indices, int f_direction = 0);
	void MUSCLEMINER_EXPORT NormalizeZeroOne(const cv::Mat& in_matrix, cv::Mat& out_matrix);
	void MUSCLEMINER_EXPORT ConvertMatToPairVector(const cv::Mat& matrix, std::vector<std::pair<int, int>>& pair_vector);
	void MUSCLEMINER_EXPORT ConvertMatToVectorOfVector(const cv::Mat& matrix, std::vector<std::vector<float>>& out_vector);
	void MUSCLEMINER_EXPORT RgbImg2Values(const cv::Mat& img, bool flag_normalize, std::vector<std::vector<float>>& values);
	void MUSCLEMINER_EXPORT Lattice(int n_rows, int n_cols, std::vector<std::pair<int, int>>& edges);
	void MUSCLEMINER_EXPORT ComputeIntensityDifferences(const std::vector<std::pair<int, int>>& edges, const std::vector<std::vector<float>>& values, cv::Mat& values_distances);
	void MUSCLEMINER_EXPORT MakeWeights(const std::vector<std::pair<int, int>>& edges, const std::vector<std::vector<float>>& values, float value_scale, float geometry_scale, cv::Mat& weights, float epsilon = 1e-5);
	void MUSCLEMINER_EXPORT AdjacencyMatrix(const std::vector<std::pair<int, int>>& edges, const cv::Mat& weights, cv::SparseMat& adjacency_matrix);
	void MUSCLEMINER_EXPORT ComputeIsoMatrixAndRowSums(const cv::Mat& img, float value_scale, cv::SparseMat& iso_matrix, cv::SparseMat& sum_rows);
	void MUSCLEMINER_EXPORT CountUnique(const std::vector<unsigned short int>& numbers, std::vector<std::pair<unsigned short int, unsigned int>>& counts);
	unsigned int MUSCLEMINER_EXPORT MaxCountUnique(const std::vector<std::pair<unsigned short int, unsigned int>>& counts, std::pair<unsigned short int, unsigned int>& max_count);
	inline bool MUSCLEMINER_EXPORT _less_by_x_for_minmax_element(const cv::Point& left, const cv::Point& right);
	inline bool MUSCLEMINER_EXPORT _less_by_y_for_minmax_element(const cv::Point& left, const cv::Point& right);
	void MUSCLEMINER_EXPORT RetrievePixels(const cv::Mat_<int>& img, const cv::Mat& mask, std::vector<int>& pixels);
	void MUSCLEMINER_EXPORT BuildTrees(const std::vector<TreeNode>& tree_nodes, std::vector<std::vector<TreeNode>>& trees);
	void MUSCLEMINER_EXPORT BuildUcmCellClassifier(const CellModel& cell_info, const cv::Mat& img, const cv::Mat& ucm, const cv::Mat& edge_map, const std::vector<float>& ucm_thresh, const float early_reject_level, std::vector<std::vector<TreeNode>>& trees, std::vector<std::vector<cv::Point>>& contour_list, std::vector<float>& node_scores);
	void MUSCLEMINER_EXPORT SegInferUcm(const CellModel& cell_info, const cv::Mat& img, const cv::Mat& ucm, const cv::Mat& edge_map, const std::vector<float>& ucm_thresh, const float early_reject_level, std::vector<std::vector<cv::Point>>& contour_list, std::vector<float>& node_scores);
	void MUSCLEMINER_EXPORT RescaleContours(std::vector<std::vector<cv::Point>>& contours, const float scale_ratio);
	void MUSCLEMINER_EXPORT DoSegmentation(const CellModel& cell_info, const cv::Mat& img, const cv::Mat& ucm, const cv::Mat& edge_map, const int n_layers, const float score_thresh, std::vector<std::vector<cv::Point>>& contours, std::vector<float>& node_scores, const float scale_ratio = 1.0);
}

#endif // MUSCLEMINER_HIERARCHICALIMAGESEGMENTATION_H_
