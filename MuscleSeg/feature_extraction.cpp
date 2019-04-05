/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   feature_extraction.cpp
Author:     Shaoju Wu

Purpose:    Extract features from single cell
******************************************************************************/


#include "feature_extraction.h"

namespace bici2
{
	FeatureExtraction::FeatureExtraction()
	{
		// std::cout << "Hello CoinCount" << std::endl;
	}


	// Obtain the orignal image
	cv::Mat FeatureExtraction::GetImage() const
	{
		return this->img_;
	}

	// Obtain the edgemap 
	cv::Mat FeatureExtraction::GetEdgeMap() const
	{
		return this->edge_map_;
	}

	// Obtain the mask region image
	cv::Mat FeatureExtraction::GetMaskRegion() const
	{
		return this->mask_region_;
	}

    //Set the parameter for initialization SparseMatrix==L, diagonal_matrix==d, img==gray(img)
    //edgemap==edgeMap maskregion==mask_region B==boundary
    void FeatureExtraction::SetParameter(cv::SparseMat& SparseMatrix, cv::SparseMat& diagonal_matrix, 
       cv::Mat& img , cv::Mat& edgemap, cv::Mat& maskregion, const std::vector<cv::Point> & Boundary)
    {
        this->sparse_matrix_ = SparseMatrix;
        this->diagonal_matrix_ = diagonal_matrix;
        this->boundary_ = Boundary;
        this->mask_region_ = maskregion;
        this->edge_map_ = edgemap;
        this->img_ = img;
    }

	// Load original image from file
	void FeatureExtraction::LoadFromIMG(std::string image_path)
	{
        cv::FileStorage yml_fs(image_path, cv::FileStorage::READ);
		//img_=cv::imread(image_path);
        yml_fs["img"] >> this->img_;
		
	}

	// Obtain the diagonal matrix of normalized cut
	cv::SparseMat FeatureExtraction::GetDiagonal() const
	{
		return this->diagonal_matrix_;
	}

	// Widen the boundary
	cv::Mat FeatureExtraction::WidenBoundary() const
	{
        cv::Mat traceB_idx;
		cv::Mat traceB = cv::Mat::zeros(4*boundary_.size(), 2, CV_32S);
        
        for (int i_row = 0; i_row < boundary_.size(); i_row++)
        {  
            traceB.at<int>(i_row, 0) = boundary_[i_row].y;
            traceB.at<int>(i_row, 1) = boundary_[i_row].x;

            traceB.at<int>(i_row + boundary_.size(), 0) = boundary_[i_row].y + 1;
            traceB.at<int>(i_row + boundary_.size(), 1) = boundary_[i_row].x;

            traceB.at<int>(i_row + 2 * boundary_.size(), 0) = boundary_[i_row].y;
            traceB.at<int>(i_row + 2 * boundary_.size(), 1) = boundary_[i_row].x + 1;

            traceB.at<int>(i_row + 3 * boundary_.size(), 0) = boundary_[i_row].y + 1;
            traceB.at<int>(i_row + 3 * boundary_.size(), 1) = boundary_[i_row].x + 1;
            
		} 

        for (int i_row = 0; i_row < traceB.rows; i_row++)
        {
            traceB_idx.push_back(img_.cols*traceB.at<int>(i_row, 0) + traceB.at<int>(i_row, 1));
        }

		return traceB_idx;
		
	}

	cv::Mat FeatureExtraction::Sub2Ind(int width, int height, std::vector<cv::Point>& Boundary)
	{
		/*sub2ind(size(a), rowsub, colsub)
		sub2ind(size(a), 2     , 3 ) = 6
		a = 1 2 3 ;
		4 5 6
		rowsub + colsub-1 * numberof rows in matrix*/
        cv::Mat index=cv::Mat::zeros(Boundary.size(),1,CV_32S);
        for (int i_row = 0; i_row < Boundary.size(); i_row++)
        {
            index.at<int>(i_row) = width*Boundary[i_row].y + Boundary[i_row].x;
        }
        return index;
	}

	// Obtain the edge pixels from the boundary
    cv::Mat  FeatureExtraction::EdgePixelObtain(const cv::Mat& linearInd)
	{
       cv::Mat edge_map = edge_map_.clone();
		cv::Mat edgePixel = cv::Mat::zeros( boundary_.size(), 1, CV_32F);
		for (int i_row = 0; i_row < boundary_.size(); i_row++)
		{
            edgePixel.at<float>(i_row, 0) = edge_map_.at<float>(linearInd.at<int>(i_row));
           // edgePixel.at<float>(i_row, 0) = edge_map_.at<float>(600, 500);
			
		}

		return edgePixel;

	}

    // Obtain the edge by using sobel detector
    cv::Mat FeatureExtraction::EdgeCompute(const cv::Mat& img)  
    {
        cv::Mat drc = img.clone();
        drc.convertTo(drc, CV_32FC1, 1.0 / 255.0);
        cv::Mat bx, by, boundary;
        cv::Sobel(drc, bx, -1, 1, 0, 3, 1.0 / 8.0);
        cv::Sobel(drc, by, -1, 0, 1, 3, 1.0 / 8.0);
        boundary = bx.mul(bx) + by.mul(by);
        cv::Scalar cutoff = cv::mean(boundary);
        float scale = 4.0;
        float edgeThreshold = std::sqrt(scale*cutoff.val[0]);
        boundary.convertTo(boundary, CV_8UC1, 255);
        cv::threshold(boundary, boundary, edgeThreshold * 255, 255, CV_THRESH_BINARY);
        //thinning(boundary, boundary);
        cvHilditchThin1(boundary, boundary);
        return boundary;

    }

    //Hilditch algorithm for thinning
    void FeatureExtraction::cvHilditchThin1(cv::Mat& src, cv::Mat& dst)
    {
        //http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html#algorithm
        //algorithm has some problems, can not get the perfect result
        if (src.type() != CV_8UC1)
        {
            printf("It can only process binary images\n");
            return;
        }
        //make a new matrix，copy src to dst
        if (dst.data != src.data)
        {
            src.copyTo(dst);
        }

        int i, j;
        int width, height;
        //to igonre the boundary problem 
        width = src.cols - 2;
        height = src.rows - 2;
        int step = src.step;
        int  p2, p3, p4, p5, p6, p7, p8, p9;
        uchar* img;
        bool ifEnd;
        int A1;
        cv::Mat tmpimg;
        while (1)
        {
            dst.copyTo(tmpimg);
            ifEnd = false;
            img = tmpimg.data + step;
            for (i = 2; i < height; i++)
            {
                img += step;
                for (j = 2; j<width; j++)
                {
                    uchar* p = img + j;
                    A1 = 0;
                    if (p[0] > 0)
                    {
                        if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01 model
                        {
                            A1++;
                        }
                        if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01model
                        {
                            A1++;
                        }
                        if (p[1] == 0 && p[step + 1]>0) //p4,p5 01model
                        {
                            A1++;
                        }
                        if (p[step + 1] == 0 && p[step]>0) //p5,p6 01model
                        {
                            A1++;
                        }
                        if (p[step] == 0 && p[step - 1]>0) //p6,p7 01model
                        {
                            A1++;
                        }
                        if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01model
                        {
                            A1++;
                        }
                        if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01model
                        {
                            A1++;
                        }
                        if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01model
                        {
                            A1++;
                        }
                        p2 = p[-step]>0 ? 1 : 0;
                        p3 = p[-step + 1]>0 ? 1 : 0;
                        p4 = p[1]>0 ? 1 : 0;
                        p5 = p[step + 1]>0 ? 1 : 0;
                        p6 = p[step]>0 ? 1 : 0;
                        p7 = p[step - 1]>0 ? 1 : 0;
                        p8 = p[-1]>0 ? 1 : 0;
                        p9 = p[-step - 1]>0 ? 1 : 0;
                        //Calculate AP2,AP4
                        int A2, A4;
                        A2 = 0;
                        //if(p[-step]>0)
                        {
                            if (p[-2 * step] == 0 && p[-2 * step + 1]>0) A2++;
                            if (p[-2 * step + 1] == 0 && p[-step + 1]>0) A2++;
                            if (p[-step + 1] == 0 && p[1]>0) A2++;
                            if (p[1] == 0 && p[0]>0) A2++;
                            if (p[0] == 0 && p[-1]>0) A2++;
                            if (p[-1] == 0 && p[-step - 1]>0) A2++;
                            if (p[-step - 1] == 0 && p[-2 * step - 1]>0) A2++;
                            if (p[-2 * step - 1] == 0 && p[-2 * step]>0) A2++;
                        }


                        A4 = 0;
                        //if(p[1]>0)
                        {
                            if (p[-step + 1] == 0 && p[-step + 2]>0) A4++;
                            if (p[-step + 2] == 0 && p[2]>0) A4++;
                            if (p[2] == 0 && p[step + 2]>0) A4++;
                            if (p[step + 2] == 0 && p[step + 1]>0) A4++;
                            if (p[step + 1] == 0 && p[step]>0) A4++;
                            if (p[step] == 0 && p[0]>0) A4++;
                            if (p[0] == 0 && p[-step]>0) A4++;
                            if (p[-step] == 0 && p[-step + 1]>0) A4++;
                        }


                        //printf("p2=%d p3=%d p4=%d p5=%d p6=%d p7=%d p8=%d p9=%d\n", p2, p3, p4, p5, p6,p7, p8, p9);
                        //printf("A1=%d A2=%d A4=%d\n", A1, A2, A4);
                        if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
                        {
                            if (((p2 == 0 || p4 == 0 || p8 == 0) || A2 != 1) && ((p2 == 0 || p4 == 0 || p6 == 0) || A4 != 1))
                            {
                                dst.at<uchar>(i, j) = 0; //If satisify the condition, set the value to 0
                                ifEnd = true;
                                //printf("\n");

                                //PrintMat(dst);
                            }
                        }
                    }
                }
            }
            //printf("\n");
            //PrintMat(dst);
            //PrintMat(dst);
            //If it didn't have any value need to set, end the loop
            if (!ifEnd) break;
        }

    }


	// Load edgeMap image, and diagonal matrix from YML file
	void FeatureExtraction::LoadFromYML(std::string yml_path)
	{
		cv::FileStorage yml_fs(yml_path, cv::FileStorage::READ);
		yml_fs["data_edge_map"] >> this->edge_map_;
		yml_fs["data_mask_region"] >> this->mask_region_;
		//yml_fs["data_diagonal_matrix"] >> this->diagonal_matrix_;
		//yml_fs["data_boundary"] >> this->boundary_;
		//yml_fs["data_sparse_matrix"] >> this->sparse_matrix_;
		yml_fs.release();
	}

	// Save cv::Mat to YML file
	void FeatureExtraction::SaveToYML(std::string yml_path) const
	{
		//cv::FileStorage yml_fs(yml_path, cv::FileStorage::WRITE);
		//yml_fs << "ucm_output" << this->img_;
		//yml_fs.release();
	}

	// display image
	void FeatureExtraction::Display(const cv::Mat& img)
	{
		cv::namedWindow("Display");

		int img_dim = img.dims;
		if (2 == img_dim)
		{
			imshow("Display", img);
		}
		else if (3 == img_dim)
		{
			int depth = img.size[0];
			int rows = img.size[1];
			int cols = img.size[2];

			cv::Mat tmp_img = cv::Mat(rows, cols, img.type());
			for (int idepth = 0; idepth < depth; ++idepth)
			{
				tmp_img.data = img.data + idepth*rows*cols*img.elemSize();
				imshow("Display", tmp_img);
				cv::waitKey(300);
				std::cout << idepth + 1 << " image." << std::endl;
			}
		}

		cv::waitKey(0);
	}

	// Calculate the centroid value (x,y) of mask region image
	void FeatureExtraction::MaskRegionCentroid(const cv::Mat& img, int& X_value, int& Y_value, int& Mask_area)
	{
		uchar BinaryVale = 255;
		Mask_area= 0;
		int cols = img.cols;
		int rows = img.rows;

		for (int i_rows = 0; i_rows < rows; i_rows++)
		{

			for (int i_cols = 0; i_cols < cols; i_cols++)
			{
				if (img.at<uchar>(i_rows, i_cols) == BinaryVale)
				{
					X_value = X_value + i_rows;
					Y_value = Y_value + i_cols;
     //count the number of white pixel within each mask region
					Mask_area++; 
				}

			}

		}
	//calculate the averaged value of (x,y)
		X_value = X_value / Mask_area;
		Y_value = Y_value / Mask_area;

	}

    //Perform one thinning iteration.
	void FeatureExtraction::thinningIteration(cv::Mat& img, int iter)
	{
		CV_Assert(img.channels() == 1);
		CV_Assert(img.depth() != sizeof(uchar));
		CV_Assert(img.rows > 3 && img.cols > 3);

		cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

		int nRows = img.rows;
		int nCols = img.cols;

		if (img.isContinuous()) {
			nCols *= nRows;
			nRows = 1;
		}

		int x, y;
		uchar *pAbove;
		uchar *pCurr;
		uchar *pBelow;
		uchar *nw, *no, *ne;    // north (pAbove)
		uchar *we, *me, *ea;
		uchar *sw, *so, *se;    // south (pBelow)

		uchar *pDst;

		// initialize row pointers
		pAbove = NULL;
		pCurr = img.ptr<uchar>(0);
		pBelow = img.ptr<uchar>(1);

		for (y = 1; y < img.rows - 1; ++y) {
			// shift the rows up by one
			pAbove = pCurr;
			pCurr = pBelow;
			pBelow = img.ptr<uchar>(y + 1);

			pDst = marker.ptr<uchar>(y);

			// initialize col pointers
			no = &(pAbove[0]);
			ne = &(pAbove[1]);
			me = &(pCurr[0]);
			ea = &(pCurr[1]);
			so = &(pBelow[0]);
			se = &(pBelow[1]);

			for (x = 1; x < img.cols - 1; ++x) {
				// shift col pointers left by one (scan left to right)
				nw = no;
				no = ne;
				ne = &(pAbove[x + 1]);
				we = me;
				me = ea;
				ea = &(pCurr[x + 1]);
				sw = so;
				so = se;
				se = &(pBelow[x + 1]);

				int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
					(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
					(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
					(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
				int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
				int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
				int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

				if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
					pDst[x] = 1;
			}
		}

		img &= ~marker;

	}


	void FeatureExtraction::thinning(const cv::Mat& src, cv::Mat& dst)
	{
		dst = src.clone();
		dst /= 255;         // convert to binary image

		cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
		cv::Mat diff;

		do {
			thinningIteration(dst, 0);
			thinningIteration(dst, 1);
			cv::absdiff(dst, prev, diff);
			dst.copyTo(prev);
		} while (cv::countNonZero(diff) > 0);

		dst *= 255;
	}

    //Calculate the properties inside the cell boundary
    void FeatureExtraction::InsideProperties(const cv::Mat& distMap, const cv::Mat& PixelCoordinate, float& insideMean, float& insideStd, cv::Mat& pixelRsps)
    {
        float max_value=0.0;
        float sum_value = 0.0;
        float sum_std = 0.0;
        int num = 0;
        //cv::Mat distValue = cv::Mat::zeros(PixelCoordinate.rows, 1, CV_64F);
        std::vector<float> distValue;
        std::vector<float>::const_iterator it_max;

        for (int i_num = 0; i_num < PixelCoordinate.rows; i_num++)
        {
            distValue.push_back(distMap.at<float>(PixelCoordinate.at<int>(i_num, 1)));
           

        }
        it_max=std::max_element(distValue.begin(), distValue.end());
        max_value=std::max(*it_max/5.0, 5.0);
        for (int i_num = 0; i_num < PixelCoordinate.rows; i_num++)
        {
            if (distMap.at<float>(PixelCoordinate.at<int>(i_num, 1))>max_value)
            {
                pixelRsps.push_back(edge_map_.at<float>(PixelCoordinate.at<int>(i_num, 1)));
                sum_value += pixelRsps.at<float>(num);
                num++;
            }

        }
        //Calculate the average value
        insideMean = sum_value / float(pixelRsps.rows);

        for (int i_num = 0; i_num < pixelRsps.rows; i_num++)
        {

            sum_std += (pixelRsps.at<float>(i_num) - insideMean)*(pixelRsps.at<float>(i_num) - insideMean);


        }
         // calculate the standard deviation
        insideStd = sqrt(sum_std / (pixelRsps.rows - 1));
    }

    // Calculate the centroid of the binary image
    void FeatureExtraction::RegionpropsCentroid(const cv::Mat& img, float& X, float& Y)
    {
        uchar BinaryVale = 255;
        
        int cols = img.cols;
        int rows = img.rows;
        float Num = 0.0;
        X = 0.0;
        Y = 0.0;

        for (int i_rows = 0; i_rows < rows; i_rows++)
        {

            for (int i_cols = 0; i_cols < cols; i_cols++)
            {
                if (img.at<uchar>(i_rows, i_cols) == BinaryVale)
                {
                    X = X + float(i_rows);
                    Y = Y + float(i_cols);
                    Num=Num+1.0;

                }

            }

        }
        //calculate the averaged value of (x,y)
        X = X / Num;
        Y = Y / Num;
    }

    //convert Mat to vector
    void FeatureExtraction::MatToVectorOfPoints(const cv::Mat& original_contour, std::vector<cv::Point>& output_contour)
    {
        for (int i_row = 0; i_row < original_contour.rows; i_row++)
        {
            cv::Point current_point(original_contour.at<int>(i_row, 1) - 1, original_contour.at<int>(i_row, 0) - 1);
            output_contour.push_back(current_point);

        }
            
    }


   //Returns a p-element vector that contains the linear indices of the pixels in the mask region.
	cv::Mat FeatureExtraction::MaskRegionpPixelIdxList(const cv::Mat& img, const int& Mask_area)
	{
		uchar BinaryVale = 255;
		int Num = 0;
		int cols = img.cols;
		int rows = img.rows;
		cv::Mat PixelIdxList = cv::Mat::zeros(Mask_area,2, CV_32S);

		for (int i_cols = 0; i_cols < cols; i_cols++)
		{

			for (int i_rows = 0; i_rows < rows; i_rows++)
			{
				if (img.at<uchar>(i_rows, i_cols) == BinaryVale)
				{
					//count the number of white pixel within each mask region
					PixelIdxList.at<int>(Num, 0) = rows*i_cols + i_rows;  // Calculate the pixel index
                   // PixelIdxList.at<int>(Num, 1) = i_rows;    // x coordinate
                  //  PixelIdxList.at<int>(Num, 2) = i_cols;    // y corrdinate
                    PixelIdxList.at<int>(Num, 1)=img_.cols*i_rows + i_cols;
					Num++;
				}

			}

		}
     
		return PixelIdxList;
	}

    //create a final feature vector
    cv::Mat FeatureExtraction::CreateFeatureVector(const float& ISO_score1, const cv::Mat& edgeHist, const float& edgeMean,
        const cv::Mat& segMean, const float& insideMean, const float& insideStd, const cv::Mat& insideHist)
    {
        //int size = 5 + edgeHist.rows + insideHist.rows;
        //cv::Mat feat = cv::Mat::zeros(size, 1, CV_32F);
        cv::Mat feat;
        feat.push_back(ISO_score1);
        feat.push_back(edgeHist.clone());
        feat.push_back(edgeMean);
        feat.push_back(segMean.clone());
        feat.push_back(insideMean);
        feat.push_back(insideStd);
        feat.push_back(insideHist.clone());

        return feat;
    }

	// obtain the range of histogram
	std::vector<float> FeatureExtraction::HistRange(const float& nBin)
	{
		float Bin = 1 / (nBin - 1);
		int Num = 1 / Bin+1;
		std::vector<float> Range;
		for (int i_Num = 0; i_Num < Num; i_Num++)
		{
			Range.push_back(Bin*i_Num);
		}

		return Range;

	}

	//separate the boudary into 6 pieces and calculate the mean value for each piece
	cv::Mat FeatureExtraction::CalculateSegMean(const cv::Mat& trace)
	{
		//separate the boundary into 6 pieces and Calculate the mean of each piece 
		int seg = std::max(1, int(boundary_.size()) / 6);
		std::vector<int> myVec;
		
		cv::Mat segmean;
		cv::Mat idx = cv::Mat::zeros(4*seg, 1, CV_32F);
		cv::Scalar tempVal;
		float segMean;
		int i_idx;
        for (int i_rows = 1; i_rows <= std::min(6, int(boundary_.size())); i_rows++)
		{
			i_idx = (i_rows - 1)*seg;

			for (int i_seg = 0; i_seg < seg; i_seg++)
			{
                idx.at<float>(i_seg) = edge_map_.at<float>(trace.at<int>(i_seg + i_idx) );
                idx.at<float>(i_seg + seg) = edge_map_.at<float>(trace.at<int>(i_seg + boundary_.size() + i_idx) );
                idx.at<float>(i_seg + 2 * seg) = edge_map_.at<float>(trace.at<int>(i_seg + 2 * boundary_.size() + i_idx) );
                idx.at<float>(i_seg + 3 * seg) = edge_map_.at<float>(trace.at<int>(i_seg + 3 * boundary_.size() + i_idx));

			}

			tempVal = cv::mean(idx);
			segmean.push_back(float(tempVal.val[0]));
			
		}
		return segmean;


	}


    //Removes interior pixels.This option sets a pixel to 0 if all its 4 - connected neighbors are 1, thus leaving only the boundary pixels on.
    cv::Mat FeatureExtraction::bwmorphRemove(const cv::Mat& img)
    {
        cv::Mat boundary = img.clone();
        int sum = 0;
           
            for (int i_row = 1; i_row < img.rows - 1; i_row++)
            {
                for (int i_col = 1; i_col < img.cols - 1; i_col++)
                {
                    if (img.at<uchar>(i_row, i_col) == 255)
                    {
                        sum = img.at<uchar>(i_row - 1, i_col) + sum;
                        sum = img.at<uchar>(i_row, i_col - 1) + sum;
                        sum = img.at<uchar>(i_row + 1, i_col) + sum;
                        sum = img.at<uchar>(i_row, i_col + 1) + sum;

                        if (sum == 1020)
                        {
                            boundary.at<uchar>(i_row, i_col) = 0;

                        }

                        sum = 0;

                          
                    }


                }


            }

        return boundary;

    }
    // Find the zero pixels from the image 
    cv::Mat FeatureExtraction::FindZeroPixel(const cv::Mat& img)
    {
        cv::Mat zero_pixel;
        for (int i_row = 0; i_row < img.rows; i_row++)
        {
            for (int i_col = 0; i_col < img.cols; i_col++)
            {
                if (img.at<uchar>(i_row,i_col)==0)
                zero_pixel.push_back(i_col + i_row*img.cols);

            }

        }

        return zero_pixel;

    }

    //Calculate the difference histogram
    cv::Mat FeatureExtraction::IntensitiesDiff(const cv::Mat& img, const cv::Mat& distance, const cv::Mat& borderPixel, const cv::Mat& distLabel)
    {
        cv::Mat Label = cv::Mat::zeros(distLabel.rows, 3, CV_32F);
        cv::Mat border;
        cv::Mat differences;
  
        border=FindZeroPixel(borderPixel);
        cv::Mat intensitiesIn = cv::Mat::zeros(border.rows, 3, CV_32F);

        for (int i_row = 0; i_row < Label.rows; i_row++)
        {
            Label.at<float>(i_row, 0) = distLabel.at<int>(i_row);
            Label.at<float>(i_row, 1) = distance.at<int>(distLabel.at<int>(i_row));
            Label.at<float>(i_row, 2) = img.at<float>(distLabel.at<int>(i_row));

        }

        for (int i_row = 0; i_row < border.rows; i_row++)
        { 
            intensitiesIn.at<float>(i_row,0) = img.at<float>(border.at<int>(i_row));
          //  intensitiesIn.at<float>(i_row, 1) = distance.at<int>(border.at<int>(i_row));
           // intensitiesIn.at<float>(i_row, 2) = Label.at<float>(intensitiesIn.at<float>(i_row, 1)-1,2);
            intensitiesIn.at<float>(i_row, 2) = Label.at<float>(distance.at<int>(border.at<int>(i_row)) - 1, 2);
            differences.push_back(std::abs(intensitiesIn.at<float>(i_row, 2) - intensitiesIn.at<float>(i_row, 0))); //Calculate the differences
        }

        return differences;

    }

    //rotate the image
    cv::Mat FeatureExtraction::rotateImage(const cv::Mat&source, double angle)
    {
        cv::Mat bordered_source;
        int top, bottom, left, right,border;
        border=std::sqrt(source.rows*source.rows + source.cols*source.cols);
        top = bottom = (border-source.rows)/2;
        left = right = (border - source.cols)/ 2;
        copyMakeBorder(source, bordered_source, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar());
        cv::Point2f src_center(bordered_source.cols / 2.0F, bordered_source.rows / 2.0F);
        cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, round(angle), 1.0);
        cv::Mat dst;
        warpAffine(bordered_source, dst, rot_mat, bordered_source.size(), cv::INTER_NEAREST);
        return dst;
    }

    // count coordinate value of white pixel in binary image
    cv::Mat FeatureExtraction::RegionpropsPixelList(const cv::Mat& img)
    {
        cv::Mat pixel_coordinate;
        cv::Mat coordinate_value = cv::Mat::zeros(1, 2, CV_32F);
        for (int i_col = 0; i_col < img.cols; i_col++)
        {
            for (int i_row = 0; i_row < img.rows; i_row++)
            {
                if (img.at<uchar>(i_row, i_col) == 255)
                {
                    coordinate_value.at <float>(0, 0) = float(i_row);
                    coordinate_value.at <float>(0, 1) = float(i_col);
                    pixel_coordinate.push_back(coordinate_value);
                }

            }

        }

        return pixel_coordinate;

    }

     //Rotationally Invariant Contour Points Distribution Histogram                                   
    cv::Mat FeatureExtraction::CPDH(const cv::Mat& img, const float& angBins, const float& radBins)
    {
        float orientation = 0.0;
        float centroid_X, centroid_Y;  // the centroid value (x,y)         
        cv::Mat polar[2];
        cv::Mat polar_value;  //polar coordinate value
        cv::Mat cartesian;  // the coordinate value index 

        orientation = RegionpropsOrientation(img);
        cv::Mat object = rotateImage(img, double(-orientation));
        orientation = RegionpropsOrientation(object);
        RegionpropsBoundingBox(object);
        //Get the object boundary only
        object = bwmorphRemove(object);
        RegionpropsCentroid(object, centroid_X, centroid_Y);
        cartesian = RegionpropsPixelList(object);


        for (int i_row = 0; i_row < cartesian.rows; i_row++)
        {
            // radius=((xi-xc)^2+(yi-yc)^2)^1/2
            polar[0].push_back(std::sqrt((cartesian.at<float>(i_row, 0) - centroid_X)*(cartesian.at<float>(i_row, 0) - centroid_X) +
                (cartesian.at<float>(i_row, 1) - centroid_Y)*(cartesian.at<float>(i_row, 1) - centroid_Y)));

            // θ=arctan((yi-yc)/(xi-xc))
            polar[1].push_back(atan2((cartesian.at<float>(i_row, 0) - centroid_X), (cartesian.at<float>(i_row, 1) - centroid_Y)));
        }

        cv::merge(polar, 2, polar_value);

        //Spatial Partitions
        double maxRo;
        float nAngBins = angBins;
        float nRadBins = radBins;
        cv::minMaxIdx(polar[0], 0, &maxRo);
        float radii = maxRo / nRadBins;
        float angles = 2 * 3.1415926 / nAngBins;
        int hist3Size[] = { int(nRadBins), int(nAngBins) };
       // float RangesRad[] = { 0.0, 11.00895, 22.0179, 33.0268, 44.0358, 55.0445 };
       // float RangesAng[] = { -3.1416, -2.618, -2.0944, -1.5708, -1.0472, -0.5236, 0, 0.5236, 1.0472, 1.5708, 2.0944, 2.618, 4.1 };
        float *RangesRad = new float[int(nRadBins) + 1];
        float *RangesAng = new float[int(nAngBins) + 1];

        for (int i_bin = 0; i_bin < int(nRadBins) + 1; i_bin++)
        {
            RangesRad[i_bin] = radii*i_bin;

        }

        if (maxRo>RangesRad[int(nRadBins)])
        {
            RangesRad[int(nRadBins)] = maxRo;

        }

        for (int i_bin = 0; i_bin < int(nAngBins) + 1; i_bin++)
        {
            RangesAng[i_bin] = -3.1415926 + angles*i_bin;

        }

        const float* ranges3[] = { RangesRad, RangesAng };
        cv::MatND hist3,histogram;
        // we compute the histogram from the 0-th and 1-st channels
        int channels3[] = { 0, 1 }; 

        cv::calcHist(&polar_value, 1, channels3, cv::Mat(), hist3, 2, hist3Size, ranges3, false, true);
        //hist3.reshape(1,60);

        for (int i_col = 0; i_col < hist3.cols; i_col++)
        {
            histogram.push_back(hist3.col(i_col));

        }
        normHist(histogram);
        //cv::transpose(histogram, histogram);

        delete[]RangesRad;
        delete[]RangesAng;

        return histogram;

    }

     //Calculate the bounding box parameter
    void FeatureExtraction::RegionpropsBoundingBox(cv::Mat& img)
    {
        std::vector< std::vector< cv::Point> > contours;
        cv::Mat contour_mask = img.clone();
        cv::findContours(contour_mask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);


        //std::vector<cv::Rect> boundRect(contours.size());
        //boundRect[0] = cv::boundingRect(contours[0]);
        //img= img(boundRect[0]);

        if (contours.size() == 1)
        {
            std::vector<cv::Rect> boundRect(contours.size());
            boundRect[0] = cv::boundingRect(contours[0]);
            img = img(boundRect[0]);
        }
        else
        {
            //find the countour that contain the biggest size
            int num_max = 0;
            for (int num_contour = 1; num_contour < contours.size(); num_contour++)
            {
                if (contours[num_max].size() < contours[num_contour].size())
                {
                    num_max = num_contour;
                }
            }

            std::vector<cv::Rect> boundRect(contours.size());
            boundRect[num_max] = cv::boundingRect(contours[num_max]);
            img = img(boundRect[num_max]);

        }


    }

    // Calculate the orientation of the binary image
    float FeatureExtraction::RegionpropsOrientation(const cv::Mat& img)
    {
        std::vector< std::vector< cv::Point> > contours_ROI;
        cv::Mat pointsf;
        cv::Mat maskROI = img.clone();
        float orientation = 0.0;
        cv::findContours(maskROI, contours_ROI, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

        std::vector<cv::Point> biggest_contour = contours_ROI[0];
        if (contours_ROI.size() > 1)
        {
            for (int i_size = 1; i_size < contours_ROI.size(); ++i_size)
            {
                if (contours_ROI[i_size].size() > biggest_contour.size())
                {
                    biggest_contour = contours_ROI[i_size];
                }
            }
        }

        cv::Mat(biggest_contour).convertTo(pointsf, CV_32F);

        //for (int i_size = 0; i_size<contours_ROI.size(); i_size++)
        //    cv::Mat(contours_ROI[i_size]).convertTo(pointsf, CV_32F);
        cv::RotatedRect box = cv::fitEllipse(pointsf);
        orientation = 90.0 - box.angle;
        return orientation;

    }

    // Normalized the histogram
    void FeatureExtraction::normalizedHist(cv::Mat& Hist)
    {
        float sum = 0.0;

        for (int i_row = 0; i_row < Hist.rows; i_row++)
        {
            sum += Hist.at<float>(i_row, 0);
        }

        for (int i_row = 0; i_row < Hist.rows; i_row++)
        {
            Hist.at<float>(i_row, 0) = Hist.at<float>(i_row, 0) / sum;
        }


    }

    //normalized histogram using norm 2
    void FeatureExtraction::normHist(cv::Mat& Hist)
    {
        float sum = 0.0;

        for (int i_row = 0; i_row < Hist.rows; i_row++)
        {
            sum += Hist.at<float>(i_row, 0)*Hist.at<float>(i_row, 0);
        }

        sum = std::sqrt(sum);

        for (int i_row = 0; i_row < Hist.rows; i_row++)
        {
            Hist.at<float>(i_row, 0) = Hist.at<float>(i_row, 0) / sum;
        }


    }

    // Increase the element of feature vector
    void FeatureExtraction::IncreaseFeatureElement(cv::Mat& featureVector, const cv::Mat& element)
    {
        for (int i_row = 0; i_row < element.rows; i_row++)
        {
            featureVector.push_back(element.at<float>(i_row));
        }

    }


	//Generate some new features generaete by using tha MICCAI 2012 paper
    std::vector<float> FeatureExtraction::EncodeFeatureMuscle()
	{
		int Mask_area = 0;
		int X_value = 0;
		int Y_value = 0;
		int threshold_value = 0;
		int threshold_type = 0;
		int nDilationScales = 2;
		int nDilations = 3;
        
        cv::Mat featureVector;

		mask_region_.convertTo(mask_region_, CV_8UC1);
		threshold(mask_region_, mask_region_, threshold_value, 255, threshold_type);
		
		// Calculate the centroid of mask region
		MaskRegionCentroid(mask_region_, X_value, Y_value,Mask_area);
		std::vector< std::vector< cv::Point> > contours;
		cv::Mat contour_mask = mask_region_.clone();
		cv::findContours(contour_mask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
		std::vector<cv::Rect> boundRect(contours.size());
		boundRect[0] = cv::boundingRect(contours[0]);
		boundRect[0].width = std::min(mask_region_.cols, boundRect[0].x + boundRect[0].width + (nDilationScales*nDilations + 2));
		boundRect[0].height = std::min(mask_region_.rows, boundRect[0].y + boundRect[0].height + (nDilationScales*nDilations + 2));
		boundRect[0].x = std::max(1, boundRect[0].x - (nDilationScales*nDilations + 2));
		boundRect[0].y = std::max(1, boundRect[0].y - (nDilationScales*nDilations + 2));
		boundRect[0].width = boundRect[0].width - boundRect[0].x;
		boundRect[0].height = boundRect[0].height - boundRect[0].y;

		cv::Mat maskROI = mask_region_(boundRect[0]);
        cv::Mat imgROI = img_(boundRect[0]);
        cv::Mat distMap;
		//Calculate the boundary
        cv::Mat boundary = EdgeCompute(mask_region_);
        boundary = 255 - boundary;
        cv::distanceTransform(boundary, distMap, cv::DIST_L2, 5);

		//compute feature
	    //recalculate the features with optimized code
		cv::Mat PixelIdxList = MaskRegionpPixelIdxList(mask_region_,Mask_area);
		int ndim = 2;
		float nomi=0.0;
		float denomi = 0.0;
		int size[] = { img_.rows*img_.cols , 1};
		// Calculate the diagonal matrix
		cv::SparseMat sm(2, size, CV_32F); 
        //create a sparse matrix for diagonal matrix
		for (int i_sparse = 0; i_sparse < Mask_area; i_sparse++)
		{
            sm.ref<float>(PixelIdxList.at<int>(i_sparse, 0), 0) = 1;
		}

	    //	ndim = nomi;
		
		for (int i_sparse = 0; i_sparse < Mask_area; i_sparse++)
		{
            nomi+= diagonal_matrix_.value<float>(PixelIdxList.at<int>(i_sparse, 0),0);

		}
		nomi=nomi/sqrt(Mask_area);
		
		// create a real sparse matrix for eigenvalue calculation
		//int size_sparse[] = { diagonal_matrix.rows, diagonal_matrix.rows };
        /*
		cv::SparseMat SparseMatrix(2, size_sparse, CV_32F);
		for (int i_sparse = 0; i_sparse < sparse_matrix_.rows; i_sparse++)
		{
			SparseMatrix.ref<float>(sparse_matrix_.at<float>(i_sparse, 0) - 1, sparse_matrix_.at<float>(i_sparse, 1) - 1) = sparse_matrix_.at<float>(i_sparse, 2);
			
		}  */

        int size_middle[] = { img_.rows*img_.cols, 1 };
		cv::SparseMat sparse_vector_output(2, size_middle, CV_32F);
		SparseVectorMatrixMultiply(sm, sparse_matrix_, sparse_vector_output);
		denomi = SparseVectorDotProduct(sparse_vector_output, sm); 
        float ISO_scorel = nomi / denomi;
	
		cv::Mat traceB = WidenBoundary();
		//boundary_.convertTo(boundary_, CV_32S);
		cv::Mat LinearInd = Sub2Ind(img_.cols, img_.rows, boundary_);
		//cv::Mat LinearIdx_traceB = Sub2Ind(img_.cols, img_.rows, traceB.col(1), traceB.col(0));
        cv::Mat EdgePixelRsps = EdgePixelObtain(LinearInd);

		// Create edge histogram based on the boundary
		int Num_nBin = 6;
		int histsize[] = { Num_nBin };
		float Ranges[] = { -0.1, 1.1 };
		int channels[] = { 0 };
		const float* ranges[] = {Ranges};
		cv::MatND edgeHist;
		//std::vector<float> nBin = HistRange(Num_nBin);
		cv::calcHist(&EdgePixelRsps, 1, channels, cv::Mat(), edgeHist, 1, histsize, ranges, true, true);
        normalizedHist(edgeHist);
	    cv::Scalar tempVal = cv::mean(EdgePixelRsps);
		float edgeMean = tempVal.val[0];
		cv::Mat segMean = CalculateSegMean(traceB);

        // Calculate the inside properties
        float insideMean, insideStd;
        cv::Mat pixelRsps;
        cv::MatND insideHist;

        InsideProperties(distMap, PixelIdxList, insideMean, insideStd, pixelRsps);
        cv::calcHist(&pixelRsps, 1, channels, cv::Mat(), insideHist, 1, histsize, ranges, true, true);
        normalizedHist(insideHist);
        cv::Mat feat = CreateFeatureVector(ISO_scorel, edgeHist, edgeMean, segMean, insideMean, insideStd, insideHist);
        IncreaseFeatureElement(featureVector, feat);

        //add difference histogram
        float nBinsDiffHist = 5;
        float dclass = 1;
        nDilationScales = 3;
        cv::Mat boundary_ROI;
        cv::Mat dilatedMask;
        cv::Mat borderBig;
        cv::Mat distanceTransf;
        cv::Mat distanceT;
        cv::Mat Zeros;
        cv::Mat differences;
        cv::MatND diffHist;
        histsize[0] = nBinsDiffHist;
        Ranges[0] = -0.1;
        Ranges[1] = dclass - dclass / nBinsDiffHist + (dclass / nBinsDiffHist)/2;

        for (int i_Dilation = 1; i_Dilation <= nDilationScales; i_Dilation++)
        {
           boundary_ROI=bwmorphRemove(maskROI);
           boundary_ROI = 255 - boundary_ROI;
           cv::dilate(maskROI, dilatedMask, cv::Mat(), cv::Point(-1, -1), i_Dilation*nDilations);
           borderBig = bwmorphRemove(dilatedMask);
           borderBig = 255 - borderBig;
           cv::distanceTransform(borderBig, distanceTransf, distanceT, cv::DIST_L2, 5, cv::DIST_LABEL_PIXEL);
           Zeros = FindZeroPixel(borderBig);
           differences = IntensitiesDiff(imgROI, distanceT, boundary_ROI, Zeros);
           cv::calcHist(&differences, 1, channels, cv::Mat(), diffHist, 1, histsize, ranges, true, true);
           normHist(diffHist);
           IncreaseFeatureElement(featureVector, diffHist);
        }

        //add shape histogram
        float nAngBins = 12.0;
        float nRadBins = 5.0;
        cv::MatND hist_CPDH;
        std::vector<float> features;
        hist_CPDH = CPDH(maskROI, nAngBins, nRadBins);
        IncreaseFeatureElement(featureVector, hist_CPDH);
        //create a feature vector using std::vector
        for (int i_size = 0; i_size < featureVector.rows; i_size++)
        {
            features.push_back(featureVector.at<float>(i_size));
        }

        return features;


	}

}