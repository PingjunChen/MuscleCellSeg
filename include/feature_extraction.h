/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   feature_extraction.h
Author:     Shaoju Wu

Purpose:    Extract features for each contour
******************************************************************************/

#ifndef MUSCLEMINER_FEATUREEXTRACTION_H_ 
#define MUSCLEMINER_FEATUREEXTRACTION_H_ 


#include <string>
#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "export.h"
#include "opencvsparsematrixmath.h"


namespace bici2
{
    class MUSCLEMINER_EXPORT FeatureExtraction
	{
	public:
		FeatureExtraction();

		// int GetCoinNumber();
        cv::SparseMat GetDiagonal() const;
		cv::Mat GetEdgeMap() const;
		cv::Mat GetImage() const;
		cv::Mat GetMaskRegion() const;
		cv::Mat WidenBoundary() const;   //widen the boundary 
        //Returns a p-element vector that contains the linear indices of the pixels in the maskregion.
		cv::Mat MaskRegionpPixelIdxList(const cv::Mat& img, const int& Mask_area);
        cv::Mat Sub2Ind(int width, int height,  std::vector<cv::Point>& Boundary);
        //separate the boudary into 6 pieces and calculate the mean value for each piece
        cv::Mat CalculateSegMean(const cv::Mat& trace);
        cv::Mat EdgePixelObtain(const cv::Mat& linearInd);  //obtain the edge pixel 
        cv::Mat EdgeCompute(const cv::Mat& img);  // Obtain the edge by using sobel detector
        //create a final feature vector
        cv::Mat CreateFeatureVector(const float& ISO_score1, const cv::Mat& edgeHist, const float& edgeMean,
            const cv::Mat& segMean, const float& insideMean, const float& insideStd, const cv::Mat& insideHist); 
        cv::Mat bwmorphRemove(const cv::Mat& img);  //Removes interior pixels. 
        cv::Mat FindZeroPixel(const cv::Mat& img);  //find the zero pixel in the binary image
        //calculate the intensity difference
        cv::Mat IntensitiesDiff(const cv::Mat& img, const cv::Mat& distance, const cv::Mat& borderPixel, const cv::Mat& distLabel); 
        cv::Mat rotateImage(const cv::Mat&source, double angle);   //Rotate the image
        cv::Mat RegionpropsPixelList(const cv::Mat& img);   // count coordinate value of white pixel in binary image
        //Rotationally Invariant Contour Points Distribution Histogram
        cv::Mat CPDH(const cv::Mat& img, const float& angBins, const float& radBins);

        void RegionpropsCentroid(const cv::Mat& img, float& X, float& Y);  // Calculate the centroid of the binary image
        void RegionpropsBoundingBox(cv::Mat& img);  //Calculate the bounding box parameter
		void LoadFromYML(std::string yml_path);     // Load YML file to OpenCV cv::Mat
		void LoadFromIMG(std::string yml_path);    //load image file to OpenCV cv::Mat
		void SaveToYML(std::string yml_path) const;       // Save cv::Mat to YML file
		void Display(const cv::Mat& img);     // Display image
        // Calculate the centroid value (x,y) of mask region
        void MaskRegionCentroid(const cv::Mat& img, int& X_value, int& Y_value, int& Mask_area);    
		void thinningIteration(cv::Mat& img, int iter);    //Perform one thinning iteration.
		void thinning(const cv::Mat& src, cv::Mat& dst);   //Function for thinning the given binary image
        //Calculate the properties inside the cell boundary
        void InsideProperties(const cv::Mat& distMap, const cv::Mat& PixelCoordinate, float& insideMean, float& insideStd, cv::Mat& pixelRsps);    
        void normalizedHist(cv::Mat& Hist);    //normalized histogram
        void normHist(cv::Mat& Hist);      //normalized histogram using norm 2
        void IncreaseFeatureElement(cv::Mat& featureVector, const cv::Mat& element);   // Increase the element of feature vector
        //Set the parameter for initialization
        void SetParameter(cv::SparseMat& SparseMatrix, cv::SparseMat& diagonal_matrix, cv::Mat& img, 
            cv::Mat& edgemap, cv::Mat& maskregion,const std::vector<cv::Point>& Boundary );
        void MatToVectorOfPoints(const cv::Mat& original_contour, std::vector<cv::Point>& output_contour);   //convert Mat to vector
        void cvHilditchThin1(cv::Mat& src, cv::Mat& dst);  //Hilditch algorithm for thinning
      
        float RegionpropsOrientation(const cv::Mat& img);   // Calculate the orientation of the binary image

		std::vector<float> HistRange(const float& nBin);   // obtain the range of histogram
        //Generate some new features generaete by using tha MICCAI 2012 paper
        std::vector<float> EncodeFeatureMuscle();
		

	private:
		cv::Mat img_;
		cv::Mat edge_map_;
		cv::Mat mask_region_;
        std::vector<cv::Point>  boundary_;
        cv::SparseMat diagonal_matrix_;
		cv::SparseMat sparse_matrix_;
	};
}

#endif // MUSCLEMINER_FEATUREEXTRACTION_H_ 