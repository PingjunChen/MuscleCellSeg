/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   treemodel.cpp
Author:     Pingjun Chen, Mason McGough

Purpose:
******************************************************************************/
#include "treemodel.h"

namespace bici2
{
	TreeModel::TreeModel(){}

	bool TreeModel::verify()
	{	// ensures that all parameters have equal length (same number of nodes in tree)
		size_t s_fids = fids.size();
		size_t s_thrs = thrs.size();
		size_t s_child = child.size();
		size_t s_distr = distr.size();
		size_t s_hs = hs.size();
		size_t s_count = count.size();
		size_t s_depth = depth.size();
		return((s_fids == s_thrs) && (s_thrs == s_child) && (s_child == s_distr) && 
			(s_distr == s_hs) && (s_hs == s_count) && (s_count == s_depth));
	}

    TreeModel::TreeModel(const TreeModel& other)
    {
        this->fids = other.fids;
        this->thrs = other.thrs;
        this->child = other.child;
        this->distr = other.distr;
        this->hs = other.hs;
        this->count = other.count;
        this->depth = other.depth;
    }

    TreeModel& TreeModel::operator=(TreeModel& other)
    {
        this->fids = other.fids;
        this->thrs = other.thrs;
        this->child = other.child;
        this->distr = other.distr;
        this->hs = other.hs;
        this->count = other.count;
        this->depth = other.depth;

        return *this;
    }

	TreeModel::~TreeModel(){}

	CellModel::CellModel(){}

    CellModel::CellModel(const CellModel& other)
    {
        this->min_area = other.min_area;
        this->max_area = other.max_area;
        this->min_solidity = other.min_solidity;
        this->cell_model = other.cell_model;
    }

    CellModel& CellModel::operator=(CellModel other)
    {
        this->min_area = other.min_area;
        this->max_area = other.max_area;
        this->min_solidity = other.min_solidity;
        this->cell_model = other.cell_model;

        return *this;
    }

    CellModel::~CellModel(){}

	RFmodel::RFmodel(){}
    // Parse model from txt file
    void RFmodel::ParseModelFromTxt(std::string path)
    {
        std::ifstream ifs(path);
        std::string content;

        const unsigned int NumFields = 7;
        // Read first line
        std::getline(ifs, content);
        // std::cout << content << std::endl;

        while (!ifs.eof())
        {
            std::getline(ifs, content);
            if (content.substr(0, 2) == "##")
            {
                TreeModel t_model;
                for (unsigned int i_field = 0; i_field < NumFields; ++i_field)
                {
                    std::getline(ifs, content);
                    if (content.substr(0, 1) == "#")
                    {
                        std::string field_name = content.substr(1);
                        if (field_name == "fids")
                        {
                            std::getline(ifs, content);
                            std::istringstream is(content);
                            t_model.fids = std::vector<unsigned int>(std::istream_iterator<unsigned int>(is),
                                std::istream_iterator<unsigned int>());
                        }
                        else if (field_name == "thrs")
                        {
                            std::getline(ifs, content);
                            std::istringstream is(content);
                            t_model.thrs = std::vector<float>(std::istream_iterator<float>(is),
                                std::istream_iterator<float>());
                        }
                        else if (field_name == "child")
                        {
                            std::getline(ifs, content);
                            std::istringstream is(content);
                            t_model.child = std::vector<unsigned int>(std::istream_iterator<unsigned int>(is),
                                std::istream_iterator<unsigned int>());
                        }
                        else if (field_name == "distr")
                        {
                            std::getline(ifs, content);
                            std::vector<float> first_ele;
                            std::istringstream is_first(content);
                            first_ele = std::vector<float>(std::istream_iterator<float>(is_first),
                                std::istream_iterator<float>());
                            std::getline(ifs, content);
                            std::vector<float> second_ele;
                            std::istringstream is_second(content);
                            second_ele = std::vector<float>(std::istream_iterator<float>(is_second),
                                std::istream_iterator<float>());
                            if (first_ele.size() != second_ele.size())
                                std::cerr << "Distr: different size Error!!!" << std::endl;
                            else
                            {
                                for (unsigned int i_distr = 0; i_distr < first_ele.size(); ++i_distr)
                                {
                                    t_model.distr.push_back(std::make_pair(first_ele[i_distr], second_ele[i_distr]));
                                }
                            }
                        }
                        else if (field_name == "hs")
                        {
                            std::getline(ifs, content);
                            std::istringstream is(content);
                            t_model.hs = std::vector<unsigned int>(std::istream_iterator<unsigned int>(is),
                                std::istream_iterator<unsigned int>());
                        }
                        else if (field_name == "count")
                        {
                            std::getline(ifs, content);
                            std::istringstream is(content);
                            t_model.count = std::vector<unsigned int>(std::istream_iterator<unsigned int>(is),
                                std::istream_iterator<unsigned int>());
                        }
                        else if (field_name == "depth")
                        {
                            std::getline(ifs, content);
                            std::istringstream is(content);
                            t_model.depth = std::vector<unsigned int>(std::istream_iterator<unsigned int>(is),
                                std::istream_iterator<unsigned int>());
                        }
                        else
                        {
                            std::cerr << "No such field!!!" << std::endl;
                            abort();
                        }
                    }
                }
                this->rf_model.push_back(t_model);
            }
            else
            {
                //std::cerr << "Parsing error!!!" << std::endl;
                //abort();
                // std::cout << "Finish Parsing" << std::endl;
                break;
            }
        }
        ifs.close();
    }
    // Dump class instance to binary file
    void RFmodel::DumpModelToBinary(std::string path)
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(reinterpret_cast<char*>(this), sizeof(*this));
        ofs.close();
    }
	RFmodel::~RFmodel(){}

	TreeNode::TreeNode(){}
	TreeNode::TreeNode(int tree_id) : tree_id{ tree_id } {}
	TreeNode::~TreeNode(){}

	TreeMax::TreeMax(){}
	TreeMax::~TreeMax(){}

	unsigned int ForestInds(const std::vector<float>& data, const std::vector<float>& thrs, const std::vector<unsigned int>& fids, const std::vector<unsigned int>& child)
	{	// assumes data is 1D
		size_t n_nodes = 1;	// original P. Dollar code uses size_t
		unsigned int idx = 0;
		while (child[idx])
		{
			if (data[fids[idx]*n_nodes] < thrs[idx]) { idx = child[idx] - 1; }
			else { idx = child[idx]; }
		}
		return(idx);
	}

	void ForestApply(const std::vector<float>& features, const std::vector<TreeModel>& forest, std::pair<float, float>& probs)
	{
		unsigned int n_features = 1;	// features is only one vector
		unsigned int n_labels = 2;		// distr is a pair 
		size_t n_trees = forest.size();
		std::pair<unsigned int, unsigned int> probs_count = std::make_pair(0, 0);
		for (unsigned int i = 0; i < n_trees; i++)
		{
			TreeModel current_tree = forest[i];
			assert(current_tree.verify());	  // all member vectors should have same size
			unsigned int current_idx = ForestInds(features, current_tree.thrs, current_tree.fids, current_tree.child);
			std::pair<unsigned int, unsigned int> current_distr = current_tree.distr[current_idx];
			probs_count = std::make_pair(probs_count.first + current_distr.first, probs_count.second + current_distr.second);
		}
		probs = std::make_pair(probs_count.first / (float)n_trees, probs_count.second / (float)n_trees);
	}

	void NodeSelection(const std::vector<TreeNode>& tree_nodes, const unsigned int n_ucm_thresholds, std::vector<int>& selected_nodes)
	{
		TreeMax blank_treemax;
		int n_tree_nodes = tree_nodes.size();
		std::vector<TreeMax> tree_max(n_tree_nodes, blank_treemax);
		for (unsigned int level = n_ucm_thresholds; level >= 1; level--)
		{
			std::vector<int> index_curr_level;
			for (unsigned int k = 0; k < tree_nodes.size(); k++)
			{
				if (level == tree_nodes[k].level)
					index_curr_level.push_back(k);
			}

			for (std::vector<int>::iterator it1 = index_curr_level.begin(); it1 != index_curr_level.end(); ++it1)
			{
				int k = *it1;
				int node_id = tree_nodes[k].node_id;
				float score_root = tree_nodes[k].score;
				float children_score = 0;
				std::vector<int> children_indxs;
				std::vector<int> node_children = tree_nodes[k].children;
				for (std::vector<int>::iterator it2 = node_children.begin(); it2 != node_children.end(); ++it2)
				{
					int child_index = *it2;
					float it_score = tree_max[child_index-1].max_sum_score;
					children_score += it_score;
					std::vector<int> it_indices = tree_max[child_index-1].max_sum_indices;
					children_indxs.insert(children_indxs.end(), it_indices.begin(), it_indices.end());
				}

				if (!children_indxs.empty() && (children_score > score_root))
				{
					tree_max[k].max_sum_score = children_score;
					tree_max[k].max_sum_indices.assign(children_indxs.begin(), children_indxs.end());
				}
				else
				{
					tree_max[k].max_sum_score = score_root;
					tree_max[k].max_sum_indices.assign(1, node_id);
				}
			}
		}
		selected_nodes.assign(tree_max[0].max_sum_indices.begin(), tree_max[0].max_sum_indices.end());
	}
}
