/******************************************************************************
Copyright:  BICI2
Created:    21:4:2016 0:26
Filename:   opencvsparsematrixmath.h
Author:     Mason McGough

Purpose:    Sparse Matrix Operation
******************************************************************************/

#ifndef MUSCLEMINER_OPENCVSPARSEMATRIXMATH_H_
#define MUSCLEMINER_OPENCVSPARSEMATRIXMATH_H_

#include <vector>

#include "export.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

namespace bici2 {

    void MUSCLEMINER_EXPORT SumSparseMatRows(const cv::SparseMat& sparse_matrix, cv::SparseMat& output);
    void MUSCLEMINER_EXPORT SparseVectorMatrixMultiply(const cv::SparseMat& vector, const cv::SparseMat& sparse_matrix, cv::SparseMat& output);
    float MUSCLEMINER_EXPORT SparseVectorDotProduct(const cv::SparseMat& vector1, const cv::SparseMat& vector2);
    void MUSCLEMINER_EXPORT AddDiagonalToSparseMat(const cv::SparseMat& sparse_matrix, const cv::SparseMat& sparse_diag, cv::SparseMat& output);
    void MUSCLEMINER_EXPORT NegativeSparseMat(const cv::SparseMat& sparse_matrix, cv::SparseMat& output);
    void MUSCLEMINER_EXPORT _CheckSparseMat(const cv::SparseMat& sparse_matrix);
}

#endif // MUSCLEMINER_OPENCVSPARSEMATRIXMATH_H_