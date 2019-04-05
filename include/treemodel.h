/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   treemodel.h
Author:     Pingjun Chen, Mason McGough

Purpose:    Load Seg model
******************************************************************************/


#ifndef MUSCLEMINER_TREEMODEL_H_
#define MUSCLEMINER_TREEMODEL_H_

#include <string>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "export.h"

namespace bici2
{
    class MUSCLEMINER_EXPORT TreeModel
    {
    public:
        TreeModel();
        TreeModel(const TreeModel& other);
        TreeModel& operator=(TreeModel& other);
        bool verify();
        ~TreeModel();
    
    public:
        std::vector<unsigned int> fids;				// feature ids
        std::vector<float>        thrs;				// thresholds
        std::vector<unsigned int> child;			// child indices
        std::vector<std::pair<float, float>> distr;	// probabilities
        std::vector<unsigned int> hs;				// likeliest label
        std::vector<unsigned int> count;			// # points at node
        std::vector<unsigned int> depth;			// depth of node
    };

    class MUSCLEMINER_EXPORT CellModel
    {
    public:
        CellModel();
        CellModel(const CellModel& other);
        CellModel& operator=(CellModel other);
        ~CellModel();

    public:
        std::vector<TreeModel> cell_model;
        int min_area;
        int max_area;
        float min_solidity;
    };

    class MUSCLEMINER_EXPORT RFmodel
    {
    public:
        RFmodel();
        void ParseModelFromTxt(std::string path);
        void DumpModelToBinary(std::string path);
        ~RFmodel();

    public:
        std::vector<TreeModel> rf_model;
    };

	class MUSCLEMINER_EXPORT TreeNode
	{
	public:
		TreeNode();
		TreeNode(int tree_id);
		~TreeNode();

	public:
		std::vector<int> children_ids;
		std::vector<int> children;
		int parent_id;
		int node_id;
		int tree_id;
		int level;
		float area;
		float threshold;
		float score;
	};

	class MUSCLEMINER_EXPORT TreeMax
	{
	public:
		TreeMax();
		~TreeMax();

	public:
		float max_sum_score;
		std::vector<int> max_sum_indices;
	};

	unsigned int MUSCLEMINER_EXPORT ForestInds(const std::vector<float>& data, const std::vector<float>& thrs, const std::vector<unsigned int>& fids, const std::vector<unsigned int>& child);
	void MUSCLEMINER_EXPORT ForestApply(const std::vector<float>& features, const std::vector<TreeModel>& forest, std::pair<float, float>& probs);
	void MUSCLEMINER_EXPORT NodeSelection(const std::vector<TreeNode>& tree_nodes, const unsigned int n_ucm_thresholds, std::vector<int>& selected_nodes);
}

#endif // MUSCLEMINER_TREEMODEL_H_