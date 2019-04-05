/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   hierarchicalimagesegmentation.cpp
Author:     Mason McGough

Purpose:    MuscleMiner implementation
******************************************************************************/

#include "hierarchicalimagesegmentation.h"

namespace bici2 {
	
	// failed to template function due to external definition
	//template <> void ExtractElements<float>(const std::vector<float>&, const std::vector<int>&, std::vector<float>&);
	void ExtractVectorInt(const std::vector<int>& original_vector, const std::vector<int>& indices, std::vector<int>& output_vector, const int offset)
	{
		output_vector.clear();
		size_t n_elements = indices.size();
		output_vector.reserve(n_elements);
		std::vector<int>::const_iterator it = indices.begin();
		std::vector<int>::const_iterator it_end = indices.end();
		for (; it != it_end; ++it)
		{
			int k = *it + offset;
			int current_element = original_vector.at(k);
			output_vector.push_back(current_element);
		}
	}
	void ExtractVectorFloat(const std::vector<float>& original_vector, const std::vector<int>& indices, std::vector<float>& output_vector, const int offset)
	{
		output_vector.clear();
		size_t n_elements = indices.size();
		output_vector.reserve(n_elements);
		std::vector<int>::const_iterator it = indices.begin();
		std::vector<int>::const_iterator it_end = indices.end();
		for (; it != it_end; ++it)
		{
			int k = *it + offset;
			float current_element = original_vector.at(k);
			output_vector.push_back(current_element);
		}
	}
	void ExtractVectorContour(const std::vector<std::vector<cv::Point>>& original_vector, const std::vector<int>& indices, std::vector<std::vector<cv::Point>>& output_vector, const int offset)
	{
		output_vector.clear();
		size_t n_elements = indices.size();
		output_vector.reserve(n_elements);
		std::vector<int>::const_iterator it = indices.begin();
		std::vector<int>::const_iterator it_end = indices.end();
		for (; it != it_end; ++it)
		{
			int k = *it + offset;
			std::vector<cv::Point> current_element = original_vector.at(k);
			output_vector.push_back(current_element);
		}
	}

	void SortIndices(const std::vector<float>& original_vector, std::vector<float>& output_vector, std::vector<int>& indices, int f_direction)
	{
		size_t n_elements = original_vector.size();
		indices.reserve(n_elements);
		for (size_t i = 0; i < n_elements; i++)
			indices.push_back(i);
		//std::iota(indices.begin(), indices.end(), 0);
		if (f_direction != 0)
			f_direction = 1;
		if (f_direction==0) // ascend
			std::sort(indices.begin(), indices.end(), [&](int i1, int i2) { return original_vector[i1] < original_vector[i2]; });
		else if (f_direction == 1)
			std::sort(indices.rbegin(), indices.rend(), [&](int i1, int i2) { return original_vector[i1] < original_vector[i2]; });
		ExtractVectorFloat(original_vector, indices, output_vector);
	}

	void NormalizeZeroOne(const cv::Mat& in_matrix, cv::Mat& out_matrix)
	{
		double min_element;
		double max_element;
		cv::minMaxLoc(in_matrix, &min_element, &max_element);
		double scale_ratio = 1.0 / (max_element - min_element);
		// eventually replace to ensure output type is same as input
		in_matrix.convertTo(out_matrix, CV_32F, scale_ratio, -scale_ratio*min_element);
	}

	void ConvertMatToPairVector(const cv::Mat& matrix, std::vector<std::pair<int, int>>& pair_vector)
	{
		int n_rows = matrix.rows;
		for (int i = 0; i < n_rows; i++)
		{
			std::pair<int, int> current_pair = std::make_pair(matrix.at<int>(i, 0), matrix.at<int>(i, 1));
			pair_vector.push_back(current_pair);
		}
	}

	void ConvertMatToVectorOfVector(const cv::Mat& matrix, std::vector<std::vector<float>>& out_vector)
	{	// assume vector is preallocated
		int n_rows = matrix.rows;
		int n_cols = matrix.cols;
		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				float current_value = matrix.at<float>(i, j);
				out_vector[i][j] = current_value;
			}
		}
	}

	void RgbImg2Values(const cv::Mat& img, bool flag_normalize, std::vector<std::vector<float>>& values)
	{
		std::vector<cv::Mat> rgb_channels(3);
		split(img, rgb_channels); //rgb_channels is vector of images
		int rows = img.rows;
		int cols = img.cols;
		int num_elements = rows*cols;
		//rgb_channels[0] = rgb_channels[0].reshape(1, 1).t();
		//rgb_channels[1] = rgb_channels[1].reshape(1, 1).t();
		//rgb_channels[2] = rgb_channels[2].reshape(1, 1).t();
		rgb_channels[0] = rgb_channels[0].t();
		rgb_channels[1] = rgb_channels[1].t();
		rgb_channels[2] = rgb_channels[2].t();
		rgb_channels[0] = rgb_channels[0].reshape(1, num_elements);
		rgb_channels[1] = rgb_channels[1].reshape(1, num_elements);
		rgb_channels[2] = rgb_channels[2].reshape(1, num_elements);
		if (flag_normalize)
		{
			NormalizeZeroOne(rgb_channels[0], rgb_channels[0]);
			NormalizeZeroOne(rgb_channels[1], rgb_channels[1]);
			NormalizeZeroOne(rgb_channels[2], rgb_channels[2]);
		}
		std::reverse(rgb_channels.begin(), rgb_channels.end());
		cv::Mat values_mat(rows, cols, CV_32FC1);
		cv::hconcat(rgb_channels, values_mat);
		ConvertMatToVectorOfVector(values_mat, values);
	}

	void Lattice(int n_rows, int n_cols, std::vector<std::pair<int, int>>& edges)
	{
		int n_pixels = n_rows*n_cols;
		std::pair<int, int> current_pair;
		for (int i = 1; i < n_pixels; i++)
		{
			if (0 != (i % n_rows))
			{
				current_pair = std::make_pair(i - 1, i);
				edges.push_back(current_pair);
			}
		}
		for (int i = n_pixels; i <= (2 * n_pixels - n_rows - 1); i++)
		{
			current_pair = std::make_pair(i - (n_pixels - 1) - 1, i - (n_pixels - 1) + n_rows - 1);
			edges.push_back(current_pair);
		}
	}

	void ComputeIntensityDifferences(const std::vector<std::pair<int, int>>& edges, const std::vector<std::vector<float>>& values, cv::Mat& values_distances)
	{
		int n_values = edges.size();
		int n_cols = values[0].size();
		for (int i = 0; i < n_values; i++)
		{
			int node1 = edges[i].first;
			int node2 = edges[i].second;
			float current_diff = 0;
			for (int j = 0; j < n_cols; j++)
				current_diff += pow(values[node1][j] - values[node2][j], 2);
			current_diff = sqrt(current_diff);
			values_distances.push_back(current_diff);
		}
		// normalize values_distances
		NormalizeZeroOne(values_distances, values_distances);
	}

	void MakeWeights(const std::vector<std::pair<int, int>>& edges, const std::vector<std::vector<float>>& values, float value_scale, float geometry_scale, cv::Mat& weights, float epsilon)
	{
		if ((value_scale > 0) && !(geometry_scale > 0))
		{
			cv::Mat values_distances;
			ComputeIntensityDifferences(edges, values, values_distances);
			weights = -(value_scale*values_distances);
			exp(weights, weights);
			weights += epsilon;
		}

		if (!(value_scale > 0) && (geometry_scale > 0))		// For full functionality compared to MATLAB code
		{

		}

		if (!(value_scale > 0) && !(geometry_scale > 0))	// For full functionality compared to MATLAB code
		{
		// exp(-(value_scale*values_distances+geometry_scale*geometry_distances))+epsilon

		}
	}

	void AdjacencyMatrix(const std::vector<std::pair<int, int>>& edges, const cv::Mat& weights, cv::SparseMat& adjacency_matrix)
	{
		int n_edges = edges.size();
		assert(n_edges == weights.rows);	// verify that vector and weights have equal number of rows
		int n_dims = 2;
		int idx[2];
		for (int i = 0; i < n_edges; i++)
		{
			idx[0] = edges[i].first;
			idx[1] = edges[i].second;
			float current_weight = weights.at<float>(i, 0);
			adjacency_matrix.ref<float>(idx) += current_weight;
			idx[0] = edges[i].second;
			idx[1] = edges[i].first;
			current_weight = weights.at<float>(i, 0);
			adjacency_matrix.ref<float>(idx) += current_weight;
		}
	}

	void ComputeIsoMatrixAndRowSums(const cv::Mat& img, float value_scale, cv::SparseMat& iso_matrix, cv::SparseMat& sum_rows)
	{
		// if img is single-channel grayscale, must convert to 3-channel img_copy
		cv::Mat img_copy(img.rows, img.cols, CV_32FC3, cv::Scalar(0, 0, 0));
		if (1 == img.channels())
		{
			cv::Mat out[] = { img_copy };
			int from_to[] = { 0, 0, 0, 1, 0, 2 };
			cv::mixChannels(&img, 1, out, 1, from_to, 3);
		}
		else { img_copy = img.clone(); }
		size_t n_rows = img_copy.rows;
		size_t n_cols = img_copy.cols;
		std::vector<std::vector<float>> values(n_rows*n_cols, std::vector<float>(3));
		RgbImg2Values(img_copy, true, values);
		std::vector<std::pair<int, int>> edges;
		Lattice(n_rows, n_cols, edges);
		float geometry_scale = 0;
		cv::Mat weights;
		MakeWeights(edges, values, value_scale, geometry_scale, weights);
		AdjacencyMatrix(edges, weights, iso_matrix);
		SumSparseMatRows(iso_matrix, sum_rows);
		NegativeSparseMat(iso_matrix, iso_matrix);
		AddDiagonalToSparseMat(iso_matrix, sum_rows, iso_matrix);
	}

	void CountUnique(const std::vector<unsigned short int>& numbers, std::vector<std::pair<unsigned short int, unsigned int>>& counts)
	{	/* Counts number of occurrences of unique elements in vector "numbers". Returns unique elements and counts
		as a pair in the vector of pairs "counts", where the first element of each pair is the unique element 
		and the second element is the number of times that element occurs in "numbers". Assumes that all elements
		in the vector numbers fall within the range [0, 65535].*/
		unsigned short int highest_element = 0;
		unsigned int count_array[USHRT_MAX] = { 0 };			// int array indicating how many of integer has been found (0 to 65535)
		bool found_array[USHRT_MAX] = { false };	// bool array indicating if integer has been found
		std::vector<unsigned short int>::const_iterator it = numbers.begin();
		std::vector<unsigned short int>::const_iterator it_end = numbers.end();
		for (; it != it_end; ++it)
		{
			unsigned short int element = *it;
			count_array[element] += 1;
			found_array[element] = (!found_array[element]) ? true : found_array[element];
			highest_element = (element>highest_element) ? element : highest_element;
		}
		for (unsigned int i = 0; i <= highest_element; i++)
		{
			if (found_array[i])
			{
				std::pair<unsigned short int, unsigned int> current_pair = std::make_pair(i, count_array[i]);
				counts.push_back(current_pair);
			}
		}
	}

	unsigned int MaxCountUnique(const std::vector<std::pair<unsigned short int, unsigned int>>& counts, std::pair<unsigned short int, unsigned int>& max_count)
	{	// Finds the element in the vector returned from CountUnique with the highest number of occurrences
		unsigned int highest_count = 0;
		size_t n_counts = counts.size();
		for (unsigned int i = 0; i < n_counts; i++)
		{
			std::pair<unsigned short int, unsigned int> current_pair = counts[i];
			if (current_pair.second > highest_count)
			{
				max_count = current_pair;
				highest_count = current_pair.second;
			}
		}
		return highest_count;
	}

	inline bool _less_by_x_for_minmax_element(const cv::Point& left, const cv::Point& right)
	{
		return left.x < right.x;
	}

	inline bool _less_by_y_for_minmax_element(const cv::Point& left, const cv::Point& right)
	{
		return left.y < right.y;
	}

	void RetrievePixels(const cv::Mat_<int>& img, const cv::Mat& mask, std::vector<unsigned short int>& pixels)
	{
		cv::SparseMat sparse_mask(mask);
		cv::SparseMatConstIterator it = sparse_mask.begin();
		cv::SparseMatConstIterator it_end = sparse_mask.end();
		for (; it != it_end; ++it)
		{
			const cv::SparseMat::Node* mat_node = it.node();
			unsigned int row_idx = mat_node->idx[0];
			unsigned int col_idx = mat_node->idx[1];
			unsigned short int current_pixel = img(row_idx, col_idx);
			pixels.push_back(current_pixel);
		}
	}

	void BuildTrees(const std::vector<TreeNode>& tree_nodes, std::vector<std::vector<TreeNode>>& trees)
	{
		size_t n_nodes = tree_nodes.size();
		for (unsigned int i = 1; i <= n_nodes; i++)
		{
			if (tree_nodes[i - 1].parent_id == 0) // root node
			{
				std::vector<TreeNode> current_tree;
				std::vector<int> q_children;
				q_children.push_back(i);
				while (!q_children.empty())
				{
					int node_id = q_children[0];
					q_children.erase(q_children.begin());
					TreeNode current_node = tree_nodes[node_id - 1];
					current_tree.push_back(current_node);
					q_children.insert(q_children.end(), current_node.children_ids.begin(), current_node.children_ids.end());
				}
				size_t n_tree = current_tree.size();
				for (unsigned int j = 1; j <= n_tree; j++)
				{
					current_tree[j - 1].children.clear();
					for (unsigned int k = 1; k <= n_tree; k++)
					{
						if (current_tree[k - 1].parent_id == current_tree[j - 1].node_id)
							current_tree[j - 1].children.push_back(k);
					}
				}
				trees.push_back(current_tree);
			}
		}
	}

	void BuildUcmCellClassifier(const CellModel& cell_info, const cv::Mat& img, const cv::Mat& ucm, const cv::Mat& edge_map, const std::vector<float>& ucm_thresh, const float early_reject_level, std::vector<std::vector<TreeNode>>& trees, std::vector<std::vector<cv::Point>>& contour_list, std::vector<float>& node_scores)
	{
		//Inputs:
		//	cellInfo <struct> :
		//		MinArea <int>
		//		MaxArea <int>(set to Inf)
		//		MinSolidity <float>
		//		cellModel <TreeModel>
		//		img <float>[520, 696, 3]
		//		ucm <float>[520, 696](range: 0 - 1)
		//		edgeMap <float>[520, 696](range: 0 - 1)
		//		goodRegionMask <float>[520, 696](unused, presumed logical)
		//		ucmThresh vector<float>[5]
		//		earlyRejectLevel <float>
		//		use_cnn_score <bool>(set to false)

		//	Outputs:
		//		trees <cell<TreeNode>>[46]
		//		nodePixList <cell<float>>[48][var](unused)
		//		contourlist <cell<float>>[48][2, var]
		//		nodeScores <float>[48]
		//		treeNodes <cell<TreeNode>>[48](unused)
		//		goodRegionMask <float>[520, 696](unused)

		// convert to gray if RGB
		cv::Mat img_gray;
		if (3 == img.channels()) { cv::cvtColor(img, img_gray, CV_BGR2GRAY); }
		else if (1 == img.channels()) { img_gray = img; }
		else { throw "Image has invalid number of channels."; }
		img_gray.convertTo(img_gray, CV_32F, 1.0/255.0, 0);	// normalize

		// assert all images are the same size
		unsigned int n_rows = img.rows;
		unsigned int n_cols = img.cols;
		assert(n_rows == ucm.rows && n_cols == ucm.cols);
		assert(n_rows == edge_map.rows && n_cols == edge_map.cols);

		float area_ratio = 0.9;
		float score_ratio = 0.0;
		float value_scale = 130.0;
		unsigned int count = 1;
		unsigned int tree_id_count = 1;
		float early_accept_level = 0.8;

		std::vector<TreeNode> tree_nodes;
		tree_nodes.reserve(64); // guess; not yet sure how large

		unsigned int n_pixels = img.rows*img.cols;
		int iso_matrix_sizes[] = { n_pixels, n_pixels };
		cv::SparseMat iso_matrix(2, iso_matrix_sizes, CV_32F);
		int sum_rows_sizes[] = { n_pixels, 1 };
		cv::SparseMat sum_rows(2, sum_rows_sizes, CV_32F);
		ComputeIsoMatrixAndRowSums(img, value_scale, iso_matrix, sum_rows);

		// preallocate mask
		cv::Mat_<int> mask_counting = cv::Mat::zeros(n_rows, n_cols, CV_16U);

		// sort ucm_thresh in descending order
		std::vector<float> ucm_thresh_copy = ucm_thresh;
		std::sort(ucm_thresh_copy.rbegin(), ucm_thresh_copy.rend());
		size_t n_thresholds = ucm_thresh_copy.size();
		cv::Point point_offset(0, 0);
		for (unsigned int i = 0; i < n_thresholds; i++)
		{
			float current_thresh = ucm_thresh_copy[i];
			cv::Mat bw_img(ucm.size(), CV_32FC1);		// threshold requires bw_img be same type as ucm
            cv::threshold(ucm, bw_img, current_thresh, 1, cv::THRESH_BINARY_INV);
			std::vector<std::vector<cv::Point>> contours;
			bw_img.convertTo(bw_img, CV_8UC1, 1, 0);	// findContours requires bw_img be 8-bit
			cv::findContours(bw_img, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, point_offset);

			size_t n_contours = contours.size();
			// storage of props probably not necessary
			for (unsigned int j = 0; j < n_contours; j++)
			{
				std::vector<cv::Point> current_contour = contours[j];
				float current_area = cv::contourArea(current_contour);
				cv::Rect current_boundingbox = cv::boundingRect(current_contour);
				std::vector<cv::Point> current_convexhull;
				cv::convexHull(current_contour, current_convexhull, false, true);
				float current_convexarea = cv::contourArea(current_convexhull);
				float current_solidity = current_area / current_convexarea;

				/* note: no contours will contain points at the limits of the image. For instance,
				in an image with 696 rows, the contours will not contain points at either 0 or 695. 
				This is a property of fillContours. */
				// skip all contours that do not meet conditions
				if (current_area < cell_info.min_area) { continue; }	// skip if area is too small
				if (current_area > cell_info.max_area) { continue; }	// skip if area is too large
				if (current_solidity < cell_info.min_solidity) { continue; }	// skip if solidity is too low
				std::pair<std::vector<cv::Point>::const_iterator, std::vector<cv::Point>::const_iterator> x_limits_it;
				x_limits_it = std::minmax_element(current_contour.begin(), current_contour.end(), _less_by_x_for_minmax_element);
				std::pair<std::vector<cv::Point>::const_iterator, std::vector<cv::Point>::const_iterator> y_limits_it;
				y_limits_it = std::minmax_element(current_contour.begin(), current_contour.end(), _less_by_y_for_minmax_element);
				std::pair<unsigned int, unsigned int> x_limits = std::make_pair((*x_limits_it.first).x, (*x_limits_it.second).x);
				std::pair<unsigned int, unsigned int> y_limits = std::make_pair((*y_limits_it.first).y, (*y_limits_it.second).y);
				if ((x_limits.first <= 1) || (x_limits.second >= n_cols - 2)) { continue; }	// skip if touching sides
				if ((y_limits.first <= 1) || (y_limits.second >= n_rows - 2)) { continue; }	// skip if touching top or bottom	

				cv::Mat current_mask = cv::Mat::zeros(n_rows, n_cols, CV_16U);
				std::vector<std::vector<cv::Point>> draw_contour;
				draw_contour.push_back(current_contour);
				cv::fillPoly(current_mask, draw_contour, cv::Scalar(1));
				std::vector<unsigned short int> pixels;
				RetrievePixels(mask_counting, current_mask, pixels);
				std::vector<std::pair<unsigned short int, unsigned int>> counts;
				CountUnique(pixels, counts);
				std::pair<unsigned short int, unsigned int> max_count;
				MaxCountUnique(counts, max_count);
				int p_id = max_count.first;
				if ((p_id > 0) && (current_area > (area_ratio * tree_nodes[p_id-1].area)))	// skip if area larger than fraction of parent
					continue;
				
				//// Feature extraction
				//FeatureExtraction feature_extraction;
				//cv::Mat edge_map_copy = edge_map.clone();
				//feature_extraction.SetParameter(iso_matrix, sum_rows, img_gray, edge_map_copy, current_mask, current_contour);
				//std::vector<float> features = feature_extraction.EncodeFeatureMuscle();
				//std::pair<float, float> probs;
				//ForestApply(features, cell_info.cell_model, probs);
				//float score = probs.second;
				//if (score < early_reject_level) { continue;	}	// skip if score too low
                
                float score = 1.0;

				// add to tree_nodes
				if (p_id == 0)
					tree_nodes.push_back(TreeNode(tree_id_count++));
				else
				{
					if (score < score_ratio*tree_nodes[p_id-1].score)	// skip if score lower than fraction of parent
						continue;
					else
					{
						tree_nodes.push_back(TreeNode());
						tree_nodes[count - 1].tree_id = tree_nodes[p_id - 1].tree_id;
					}
				}
				tree_nodes[count-1].threshold = current_thresh;
				tree_nodes[count-1].node_id = count;
				tree_nodes[count-1].parent_id = p_id;
				tree_nodes[count-1].area = current_area;
				tree_nodes[count-1].score = score;
				node_scores.push_back(tree_nodes[count-1].score);
				contour_list.push_back(current_contour);
				if (p_id > 0)
				{
					tree_nodes[p_id-1].children_ids.push_back(count);
					tree_nodes[count-1].level = tree_nodes[p_id-1].level + 1;
				}
				else
					tree_nodes[count-1].level = 1;
				// add contour to label matrix mask_counting
				cv::fillPoly(mask_counting, draw_contour, cv::Scalar(count));
				count++;
			}
		}
		BuildTrees(tree_nodes, trees);
	}

	void SegInferUcm(const CellModel& cell_info, const cv::Mat& img, const cv::Mat& ucm, const cv::Mat& edge_map, const std::vector<float>& ucm_thresh, const float early_reject_level, std::vector<std::vector<cv::Point>>& contour_list, std::vector<float>& node_scores)
	{
		std::vector<std::vector<TreeNode>> trees;
		trees.reserve(64);	// guess, not sure yet how large
		std::vector<std::vector<cv::Point>> contour_list_temp;
		std::vector<float> node_scores_temp;
		BuildUcmCellClassifier(cell_info, img, ucm, edge_map, ucm_thresh, early_reject_level, trees, contour_list_temp, node_scores_temp);
		size_t n_threshes = ucm_thresh.size();
		size_t n_trees = trees.size();
		std::vector<int> select_indices;
		for (size_t i = 0; i < n_trees; i++)	// find indices of contours to keep
		{
			std::vector<int> new_nodes;
			std::vector<TreeNode> current_tree_nodes = trees[i];
			NodeSelection(current_tree_nodes, n_threshes, new_nodes);
			select_indices.insert(select_indices.end(), new_nodes.begin(), new_nodes.end());
		}
		// remove unselected contours and scores
		std::vector<float> node_scores_temp2;
		ExtractVectorFloat(node_scores_temp, select_indices, node_scores_temp2, -1);
		std::vector<int> node_indices;
		SortIndices(node_scores_temp2, node_scores, node_indices, bici2::SORT_DESCEND);
		std::vector<int> sorted_select_indices;
		ExtractVectorInt(select_indices, node_indices, sorted_select_indices, 0);
		ExtractVectorContour(contour_list_temp, sorted_select_indices, contour_list, -1);
	}

	void RescaleContours(std::vector<std::vector<cv::Point>>& contours, const float scale_ratio)
	{	/* Resizes the points in contours by scale_ratio, a scaling factor. */
		size_t n_contours = contours.size();
		for (size_t i = 0; i < n_contours; i++)
		{
			size_t n_points = contours[i].size();
			for (size_t j = 0; j < n_points; j++)
			{
				int current_x = round(contours[i][j].x * scale_ratio);
				int current_y = round(contours[i][j].y * scale_ratio);
				contours[i][j] = cv::Point(current_x, current_y);
			}
		}
	}

	void DoSegmentation(const CellModel& cell_info, const cv::Mat& img, const cv::Mat& ucm, const cv::Mat& edge_map, int n_layers, const float score_thresh, std::vector<std::vector<cv::Point>>& contours, std::vector<float>& node_scores, const float scale_ratio)
	{
		if (n_layers > 5)	// only five elements in possible_threshes
			n_layers = 5;
		std::vector<float> possible_threshes = { (float) 0.99, (float) 0.95, (float) 0.9, (float) 0.8, (float) 0.7 };
		std::vector<float> ucm_thresh_vector;
		ucm_thresh_vector.assign(possible_threshes.begin(), possible_threshes.begin() + n_layers);
		float early_reject_level = 0.01;
		std::vector<std::vector<cv::Point>> contour_list_temp;
		std::vector<float> node_scores_temp;
		bici2::SegInferUcm(cell_info, img, ucm, edge_map, ucm_thresh_vector, early_reject_level, contour_list_temp, node_scores_temp);
		size_t n_scores = node_scores_temp.size();
		std::vector<int> nodes_to_keep;
		nodes_to_keep.reserve(n_scores);
		for (size_t i = 0; i < n_scores; i++)	// keep contours with scores above threshold
		{
			if (node_scores_temp[i]>score_thresh)
				nodes_to_keep.push_back(i);
		}
		bici2::ExtractVectorFloat(node_scores_temp, nodes_to_keep, node_scores);
		bici2::ExtractVectorContour(contour_list_temp, nodes_to_keep, contours);
		if (scale_ratio != (float) 1.0)	// rescale contours if needed
			bici2::RescaleContours(contours, scale_ratio);
	}
}
