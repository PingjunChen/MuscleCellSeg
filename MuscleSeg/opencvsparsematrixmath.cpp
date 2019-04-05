/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   opencvsparsematrixmath.cpp
Author:     Mason McGough

Purpose:
******************************************************************************/

#include "opencvsparsematrixmath.h"

namespace bici2 {

	void SumSparseMatRows(const cv::SparseMat& sparse_matrix, cv::SparseMat& output)
	{	// Sums across the rows of an NxM sparse matrix, returning an Nx1 sparse matrix of sums
		cv::SparseMatConstIterator it = sparse_matrix.begin();
		cv::SparseMatConstIterator it_end = sparse_matrix.end();
		for (; it != it_end; ++it)
		{
			const cv::SparseMat::Node* mat_node = it.node();
			int row_idx = mat_node->idx[0];
			float current_value = it.value<float>();
			output.ref<float>(row_idx, 0) += current_value;
			if (0 == output.value<float>(row_idx, 0)) // have to manually remove values that become zero
			{
				output.erase(row_idx, 0);
			}
			//delete mat_node;
		}
	}

	void SparseVectorMatrixMultiply(const cv::SparseMat& vector, const cv::SparseMat& sparse_matrix, cv::SparseMat& output)
	{	// Multiplies an Nx1 sparse vector with an NxM sparse matrix (of the form vector*sparse_matrix). 
		// output is same size as vector
		// left multiply matrix with vector (v'*A)
		// for elements of matrix:
		//		retrieve row_idx, col_idx, and element value
		//		retrieve element in vector with same row_idx using .value(row_idx) [returns 0 if does not exist]:
		//		multiply vector(row_idx)*matrix(row_idx,col_idx)
		//		add to output(col_idx)
		assert(vector.size(0) == sparse_matrix.size(0));	// verify that vector and matrix have equal number of rows
		assert(vector.size(0) == output.size(0));			// verify that vector and output have equal number of rows
		assert(0 == output.nzcount());	// ensure that output is empty before beginning
		cv::SparseMatConstIterator it = sparse_matrix.begin();
		cv::SparseMatConstIterator it_end = sparse_matrix.end();
		for (; it != it_end; ++it)
		{
			const cv::SparseMat::Node* mat_node = it.node();
			int row_idx = mat_node->idx[0];
			int col_idx = mat_node->idx[1];
			float prod1 = vector.value<float>(row_idx, 0);
			float prod2 = it.value<float>();
			output.ref<float>(col_idx, 0) += prod1*prod2;
			if (0 == output.value<float>(col_idx, 0)) // have to manually remove values that become zero
			{
				output.erase(col_idx, 0);
			}
			//delete mat_node;
		}
	}

	float SparseVectorDotProduct(const cv::SparseMat& vector1, const cv::SparseMat& vector2)
	{	// Performs the dot product between two Nx1 vectors vector1 and vector2. 
		assert(vector1.size(0) == vector2.size(0));	// verify that vectors have equal number of rows
		// Be sure that output is initialized to zero
		cv::SparseMatConstIterator it = vector1.begin();
		cv::SparseMatConstIterator it_end = vector1.end();
		float output = 0.0;
		for (; it != it_end; ++it)
		{
			const cv::SparseMat::Node* mat_node = it.node();
			int col_idx = mat_node->idx[0];
			float prod1 = it.value<float>();
			float prod2 = vector2.value<float>(col_idx, 0);
			output += prod1*prod2;
			//delete mat_node;
		}
		return output;
	}

	void AddDiagonalToSparseMat(const cv::SparseMat& sparse_matrix, const cv::SparseMat& sparse_diag, cv::SparseMat& output)
	{	// Adds elements of Nx1 vector sparse_diag to diagonal of an NxN sparse_matrix.
		assert(sparse_matrix.size(0) == sparse_matrix.size(1));	// verify sparse_matrix is square
		assert(sparse_matrix.size(0) == sparse_diag.size(0));	// verify sparse_matrix and sparse_diag have same # rows
		output = sparse_matrix;
		cv::SparseMatConstIterator it = sparse_diag.begin();
		cv::SparseMatConstIterator it_end = sparse_diag.end();
 		for (; it != it_end; ++it)
		{
			const cv::SparseMat::Node* mat_node = it.node();
			int row_idx = mat_node->idx[0];
			output.ref<float>(row_idx, row_idx) += it.value<float>();
			//float current_value = it.value<float>();
			//output.ref<float>(col_idx, col_idx) -= current_value;
			if (0 == output.value<float>(row_idx, row_idx)) // have to manually remove values that become zero
			{
				output.erase(row_idx, row_idx);
			}
			//delete mat_node;
		}
	}

	void NegativeSparseMat(const cv::SparseMat& sparse_matrix, cv::SparseMat& output)
	{	// Makes all elements of sparse_matrix negative.
		cv::SparseMatConstIterator it = sparse_matrix.begin();
		cv::SparseMatConstIterator it_end = sparse_matrix.end();
		for (; it != it_end; ++it)
		{
			const cv::SparseMat::Node* mat_node = it.node();
			int row_idx = mat_node->idx[0];
			int col_idx = mat_node->idx[1];
			output.ref<float>(row_idx, col_idx) = -it.value<float>();
			//delete mat_node;
		}
	}

	void _CheckSparseMat(const cv::SparseMat& sparse_matrix)
	{	// used for debugging purposes
		cv::SparseMatConstIterator it = sparse_matrix.begin();
		cv::SparseMatConstIterator it_end = sparse_matrix.end();
		for (; it != it_end; ++it)
		{
			const cv::SparseMat::Node* mat_node = it.node();
			int row_idx = mat_node->idx[0];
			int col_idx = mat_node->idx[1];
			float current_value = it.value<float>();
			int b = 0;
			//delete mat_node;
		}
	}
}