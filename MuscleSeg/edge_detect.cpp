#include "edge_detect.h"

namespace bici2
{
    #define pi = 3.1416;

	EdgeDetect::EdgeDetect()
	{}

	void EdgeDetect::SetImg(const cv::Mat& img)
	{
		this->img_ = img.clone();
	}

    void EdgeDetect::SetModel(const std::string& opts_path,
        const std::string& thrs_path,
        const std::string& fids_path,
        const std::string& child_path,
        const std::string& count_path,
        const std::string& depth_path,
        const std::string& ebins_path,
        const std::string& ebnds_path)
    {
        //this->model_.ParseModelFromTxt(model_path);
        this->model_.ParseEdgeModel(opts_path, thrs_path, fids_path, child_path, count_path,
                                    depth_path, ebins_path, ebnds_path);
        this->model_.opts.nTreesEval = std::min(model_.opts.nTreesEval, model_.opts.nTrees);
        this->model_.opts.stride = std::max(this->model_.opts.stride, this->model_.opts.shrink);
    }

    cv::Mat EdgeDetect::GetRandomForestResult(const cv::Mat& ori_img)
    {
        cv::Mat convertedImg;
        ori_img.convertTo(convertedImg, CV_32FC3, 1.0 / 255.0);

        clock_t begin, end;
        begin = clock();
        std::cout << "Start Edge Detection ... \n";
        cv::Mat edge_map = EdgeDetectfunction(convertedImg);
        end = clock();
        std::cout << "Edge Detection Finished ... \n";
        std::cout << "Edge Detection takes " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

        double min_val = 0;
        double max_val = 0;
        cv::minMaxLoc(edge_map, &min_val, &max_val);
        return edge_map * (255.0 / max_val);
    }

    cv::Mat EdgeDetect::EdgeDetectfunction(const cv::Mat& img)
    {
        //cv::Mat E;
        cv::Mat Es = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

        float ss[3] = { 0.5, 1.0, 2.0 };
        int k = sizeof(ss) / sizeof(ss[0]);
        std::vector<int> siz = { img.rows, img.cols, img.channels() };  // store img_ size as (cols, rows, channels);
        this->model_.opts.multiscale = 0;
        this->model_.opts.nms = 0;

        for (size_t i = 0; i < k; i++)
        {
            float s = ss[i];
            cv::Mat I1 = ImResample(img, s);
            cv::Mat Es1 = mutlipleCalculate(I1);
            std::vector<int> subSiz;
            subSiz.push_back(siz.at(0));
            subSiz.push_back(siz.at(1));
            cv::Mat E2 = ImResample(Es1, subSiz);
            Es = Es + E2;
        }
        Es = Es / 3.0;
        //compute E and O and perform nms
        //E = Es;
        return Es;
    }

    cv::Mat EdgeDetect::mutlipleCalculate(const cv::Mat& img){
        //cv::Mat E;
        cv::Mat Es = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

        std::vector<int> sizOrig = { img.rows, img.cols, img.channels() };
        int r = this->model_.opts.imWidth / 2;
        std::vector<int> p = { r, r, r, r };
        p.at(1) = p.at(1) + std::fmod(4 - std::fmod(sizOrig[0] + 2 * r, 4), 4);
        p.at(3) = p.at(3) + std::fmod(4 - std::fmod(sizOrig[1] + 2 * r, 4), 4);
        std::string type = "symmetric";
        cv::Mat img_tmp = imPad(img, p, type);

        //computer features and apply forest to image
        edgeChns(img_tmp, this->model_.opts);
        /*cv::Mat Test1 = cv::Mat::zeros(146, 190, CV_32F);
        std::memcpy(Test1.data, this->chnsSim.data + 0 * 146 * 190 * this->chnsSim.elemSize1(), 146 * 190 * this->chnsSim.elemSize1());*/

#pragma region Test final result by reading chnsReg, chnsSim from matlab;
        //if (times == 0)
        //{
        //    cv::Mat chnsReg[13];
        //    for (int i = 1; i <= 13; i++)
        //    {
        //        std::string path = "..\\..\\..\\Data\\chnsReg1_" + std::to_string(i) + ".yml";
        //        bici2::YmlRead ymlread1(path);
        //        ymlread1.Image(chnsReg[i - 1]);
        //    }
        //    cv::Mat chnsSim[13];
        //    for (int i = 1; i <= 13; i++)
        //    {
        //        std::string path = "..\\..\\..\\Data\\chnsSim1_" + std::to_string(i) + ".yml";
        //        bici2::YmlRead ymlread1(path);
        //        ymlread1.Image(chnsSim[i - 1]);
        //    }

        //    //cv::Mat Test1 = cv::Mat::zeros(146, 190, CV_32F);
        //    //std::memcpy(Test1.data, chnsReg[1].data, 146 * 190 * Test1.elemSize1());

        //    int chns_size[3] = { 13, 242, 448 };
        //    cv::Mat chnsReg_ = cv::Mat::zeros(3, chns_size, CV_32F);
        //    cv::Mat chnsSim_ = cv::Mat::zeros(3, chns_size, CV_32F);

        //    for (size_t i = 0; i < 13; i++)
        //    {
        //        std::memcpy(chnsReg_.data + i * 242 * 448 * chnsReg_.elemSize1(), chnsReg[i].data, 242 * 448 * chnsReg_.elemSize1());
        //        std::memcpy(chnsSim_.data + i * 242 * 448 * chnsSim_.elemSize1(), chnsSim[i].data, 242 * 448 * chnsSim_.elemSize1());
        //    }
        //    //cv::Mat Test1 = cv::Mat::zeros(146, 190, CV_32F);
        //    //std::memcpy(Test1.data, chnsReg_.data + 0 * 146 * 190 * chnsReg_.elemSize1(), 146 * 190 * chnsReg_.elemSize1());
        //    //cv::Mat Test2 = cv::Mat::zeros(146, 190, CV_32F);
        //    //std::memcpy(Test2.data, chnsSim_.data + 0 * 146 * 190 * chnsSim_.elemSize1(), 146 * 190 * chnsSim_.elemSize1());
        //    //success
        //    times++;
        //    Es = EdgeDetectMex(this->model_, img, chnsReg_, chnsSim_);
        //}
        //else if (times == 1)
        //{
        //    cv::Mat chnsReg[13];
        //    for (int i = 1; i <= 13; i++)
        //    {
        //        std::string path = "..\\..\\..\\Data\\chnsReg2_" + std::to_string(i) + ".yml";
        //        bici2::YmlRead ymlread1(path);
        //        ymlread1.Image(chnsReg[i - 1]);
        //    }
        //    cv::Mat chnsSim[13];
        //    for (int i = 1; i <= 13; i++)
        //    {
        //        std::string path = "..\\..\\..\\Data\\chnsSim2_" + std::to_string(i) + ".yml";
        //        bici2::YmlRead ymlread1(path);
        //        ymlread1.Image(chnsSim[i - 1]);
        //    }

        //    //cv::Mat Test1 = cv::Mat::zeros(146, 190, CV_32F);
        //    //std::memcpy(Test1.data, chnsReg[1].data, 146 * 190 * Test1.elemSize1());

        //    int chns_size[3] = { 13, 468, 880 };
        //    cv::Mat chnsReg_ = cv::Mat::zeros(3, chns_size, CV_32F);
        //    cv::Mat chnsSim_ = cv::Mat::zeros(3, chns_size, CV_32F);

        //    for (size_t i = 0; i < 13; i++)
        //    {
        //        std::memcpy(chnsReg_.data + i * 468 * 880 * chnsReg_.elemSize1(), chnsReg[i].data, 468 * 880 * chnsReg_.elemSize1());
        //        std::memcpy(chnsSim_.data + i * 468 * 880 * chnsSim_.elemSize1(), chnsSim[i].data, 468 * 880 * chnsSim_.elemSize1());
        //    }
        //    times++;
        //    Es = EdgeDetectMex(this->model_, img, chnsReg_, chnsSim_);
        //}
        //else if ( times == 2)
        //{
        //    cv::Mat chnsReg[13];
        //    for (int i = 1; i <= 13; i++)
        //    {
        //        std::string path = "..\\..\\..\\Data\\chnsReg3_" + std::to_string(i) + ".yml";
        //        bici2::YmlRead ymlread1(path);
        //        ymlread1.Image(chnsReg[i - 1]);
        //    }
        //    cv::Mat chnsSim[13];
        //    for (int i = 1; i <= 13; i++)
        //    {
        //        std::string path = "..\\..\\..\\Data\\chnsSim3_" + std::to_string(i) + ".yml";
        //        bici2::YmlRead ymlread1(path);
        //        ymlread1.Image(chnsSim[i - 1]);
        //    }

        //    //cv::Mat Test1 = cv::Mat::zeros(146, 190, CV_32F);
        //    //std::memcpy(Test1.data, chnsReg[1].data, 146 * 190 * Test1.elemSize1());

        //    int chns_size[3] = { 13, 920, 1742 };
        //    cv::Mat chnsReg_ = cv::Mat::zeros(3, chns_size, CV_32F);
        //    cv::Mat chnsSim_ = cv::Mat::zeros(3, chns_size, CV_32F);

        //    for (size_t i = 0; i < 13; i++)
        //    {
        //        std::memcpy(chnsReg_.data + i * 920 * 1742 * chnsReg_.elemSize1(), chnsReg[i].data, 920 * 1742 * chnsReg_.elemSize1());
        //        std::memcpy(chnsSim_.data + i * 920 * 1742 * chnsSim_.elemSize1(), chnsSim[i].data, 920 * 1742 * chnsSim_.elemSize1());
        //    }
        //    Es = EdgeDetectMex(this->model_, img, chnsReg_, chnsSim_);
        //}
#pragma endregion
        Es = EdgeDetectMex(this->model_, img_tmp, this->chnsReg, this->chnsSim);

        //normalize and finalize edge maps
        //float t = 2 * this->model_.opts.stride * this->model_.opts.stride / (this->model_.opts.gtWidth * this->model_.opts.gtWidth * this->model_.opts.nTreesEval );
        float t = 0.00781250;
        r = this->model_.opts.gtWidth / 2;
        cv::Rect roi1 = cv::Rect(r, r, sizOrig[1], sizOrig[0]);
        cv::Mat subImg;
        subImg = Es(roi1) * t;
        Es = convTri(subImg, 1);

        //E = Es;
        return Es;
    }

    cv::Mat EdgeDetect::ImResample(const cv::Mat& img, float scale)
	{
        if (img.dims > 2)
        {
            return img;
        }
        int imgCols = img.cols;
        int imgRows = img.rows;
		int bilinear = 1;
		int norm = 1;
		int dimsOfImg = 1;
		cv::Mat result;

		bool same = (dimsOfImg == 1 && scale == 1);
		if (same && norm == 1)
		{
            result = img;
            return result;
		}
		if (bilinear == 1)
		{
			int newCols;
			int newRows;
			if (dimsOfImg == 1)
			{
				newCols = std::round(scale* imgCols);
				newRows = std::round(scale* imgRows);
			}
            result = bici2::ImResampleMex(img, newRows, newCols, norm);
            return result;
		}
	}

    cv::Mat EdgeDetect::ImResample(const cv::Mat& img, std::vector<int> scale)
	{
		int bilinear = 1;
		int norm = 1;
		int imgCols = img.cols;
		int imgRows = img.rows;
		int newCols, newRows;
		int dimsOfImg = scale.size();
		cv::Mat result;

		bool same = (dimsOfImg == 2 && imgCols == scale[0] && imgRows == scale[1]);
		if (same && norm == 1)
		{
			result = img;
			return result;
		}
		if (bilinear)
		{
			newCols = scale[0];
			newRows = scale[1];
            result = bici2::ImResampleMex(img, newCols, newRows, norm);
            return result;
		} 
	}

    cv::Mat EdgeDetect::imPad(const cv::Mat& I, std::vector<int> pad, std::string type)
	{
		return bici2::ImPadMex(I, pad, type);
	}

    void EdgeDetect::edgeChns(const cv::Mat& I, EdgeModelOpts opts)
    {
		int shrink = opts.shrink;
		int nTypes = 1;
		int k = 0;
		int chnSm;
		int simSm;
		std::string cs;
		cv::Mat Ishrink;
		cv::Mat I1;
		cv::Mat H;
        cv::Mat img;
        int chns_size[3] = { 13, (I.rows + 1)/shrink, (I.cols +1 )/shrink };
        cv::Mat chns = cv::Mat::zeros(3, chns_size, CV_32F);
		for (size_t i = 1; i <= nTypes; i++)
		{
			if (I.channels() == 1)
			{
				cs = "gray";
			}
			else
			{
				cs = "luv";
			}
            img = RbgConvert(I, cs);
            Ishrink = EdgeDetect::ImResample(img, (float)1 / shrink);
            //std::memcpy(chns.data, Ishrink.data, 3* Ishrink.rows*Ishrink.cols*Ishrink.elemSize1());

            cv::Mat split_chan[3];
            cv::split(Ishrink, split_chan);
            std::memcpy(chns.data + k * Ishrink.rows * Ishrink.cols * Ishrink.elemSize1(), split_chan[2].data, Ishrink.rows * Ishrink.cols * Ishrink.elemSize1());
            k++;
            std::memcpy(chns.data + k * Ishrink.rows * Ishrink.cols * Ishrink.elemSize1(), split_chan[1].data, Ishrink.rows * Ishrink.cols * Ishrink.elemSize1());
            k++;
            std::memcpy(chns.data + k * Ishrink.rows * Ishrink.cols * Ishrink.elemSize1(), split_chan[0].data, Ishrink.rows * Ishrink.cols * Ishrink.elemSize1());
            k++;
            /*cv::Mat Test1 = cv::Mat(Ishrink.rows, Ishrink.cols, CV_32F);
            std::memcpy(Test1.data, chns.data + 2*Ishrink.rows*Ishrink.cols*Ishrink.elemSize1(), Ishrink.rows*Ishrink.cols*Ishrink.elemSize1());*/

			for (size_t i = 1; i <= 2; i++)
			{
                int s = std::pow(2, i - 1);
				if (s == shrink)
				{
					I1 = Ishrink;
				}
				else
				{
                    I1 = EdgeDetect::ImResample(img, (float)1 / s);
				}
				I1 = convTri(I1, opts.grdSmooth);
                cv::Mat MO = GradientMag(I1, 0, opts.normRad, 0.01); // generate this->M and this->O;
                cv::Mat chans[2];
                cv::split(MO, chans);
                this->M = chans[0];
                this->O = chans[1];
                cv::Mat temp_M = this->M;
                cv::Mat temp_O = this->O;
                H = GradientHist(this->M, this->O, std::max(1, shrink / s), opts.nOrients, 0);
                
                H = bici2::EdgeDetect::ImResample(H, std::max(1, s/shrink));

                this->M = bici2::EdgeDetect::ImResample(this->M, (float)s / shrink);
                temp_M = this->M;
                std::memcpy(chns.data + k* Ishrink.rows*Ishrink.cols*Ishrink.elemSize1(), temp_M.data, Ishrink.rows*Ishrink.cols*Ishrink.elemSize1());
                k = k + 1; 
                std::memcpy(chns.data + k* Ishrink.rows*Ishrink.cols*Ishrink.elemSize1(), H.data, 4 * Ishrink.rows*Ishrink.cols*Ishrink.elemSize1());
				k = k + 4;
			}
		}
        // chns has no problem 
        

		chnSm = opts.chnSmooth / shrink;
		if (chnSm > 1)
		{
			chnSm = (int) std::round(chnSm);
		}
		simSm = opts.simSmooth / shrink;
		if (simSm > 1)
		{
			simSm = (int) std::round(simSm);
		}
        chnsReg = cv::Mat::zeros(3, chns_size, CV_32F);
        chnsSim = cv::Mat::zeros(3, chns_size, CV_32F);
		chnsReg = convTri(chns, chnSm);
		chnsSim = convTri(chns, simSm);


       /* cv::Mat Test1 = cv::Mat(Ishrink.rows, Ishrink.cols, CV_32F);
        std::memcpy(Test1.data, chnsReg.data + 9 * Ishrink.rows*Ishrink.cols*Ishrink.elemSize1(), Ishrink.rows*Ishrink.cols*Ishrink.elemSize1());
        cv::Mat Test2 = cv::Mat(Ishrink.rows, Ishrink.cols, CV_32F);
        std::memcpy(Test2.data, chnsReg.data + 10 * Ishrink.rows*Ishrink.cols*Ishrink.elemSize1(), Ishrink.rows*Ishrink.cols*Ishrink.elemSize1());
        cv::Mat Test3 = cv::Mat(Ishrink.rows, Ishrink.cols, CV_32F);
        std::memcpy(Test3.data, chnsReg.data + 11* Ishrink.rows*Ishrink.cols*Ishrink.elemSize1(), Ishrink.rows*Ishrink.cols*Ishrink.elemSize1());
        cv::Mat Test4 = cv::Mat(Ishrink.rows, Ishrink.cols, CV_32F);
        std::memcpy(Test4.data, chnsReg.data + 12* Ishrink.rows*Ishrink.cols*Ishrink.elemSize1(), Ishrink.rows*Ishrink.cols*Ishrink.elemSize1());*/
        
       // chnsReg, chnsSim both are correct
        this->chnsReg = chnsReg;
        this->chnsSim = chnsSim;
	}

    cv::Mat EdgeDetect::convTri(const cv::Mat& img, int radius)
	{
		int s = 1;
		int nomex = 0;
		int m;
        int imgRows;
        int imgCols;
		cv::Mat resultImg;
		if (img.empty() || (radius == 0 && s == 1)) // make sure 3 channel return directly.
		{
			return resultImg = img;
		}
        if (img.dims == 2)
        {
             imgRows = img.rows;
             imgCols = img.cols;
        }
        else
        {
            imgCols = img.size[2]; // for 13 dims img
            imgRows = img.size[1]; // for 13 dims img
        }
		m = std::min(imgRows,imgCols);
		if (m < 4 || 2*radius + 1 > m )
		{
			nomex = 1;
		}

		if (nomex == 0)
		{
			if (radius > 0 && radius <= 1 && s <=2)
			{
                resultImg = bici2::convConstMex("convTri1", img, 12 / radius / (radius + 2) - 2, s);
				return resultImg;
			}
			else
			{
				resultImg = bici2::convConstMex("convTri", img, radius, s);
				return resultImg;
			}
		}
	}

    cv::Mat EdgeDetect::RbgConvert(const cv::Mat& inputImg, std::string colorSpace)
	{
		bool useSingle = true;
		bool norm;
		int channel;
		bool flag;
        int flag_;
		cv::Mat outputImg;
		std::string outClass;
		std::vector<std::string> colorSpace_ = { "gray", "rgb", "luv", "hsv", "orig" };
		flag = std::find(colorSpace_.begin(), colorSpace_.end(), colorSpace) != colorSpace_.end();
        if (colorSpace == "gray")
        {
            flag_ = 0;
        }
        else if (colorSpace == "rgb")
        {
            flag_ = 1;
        }
        else if (colorSpace == "luv")
        {
            flag_ = 2;
        }
        else if (colorSpace == "hsv")
        {
            flag_ = 3;
        }
        else if (colorSpace == "orig")
        {
            flag_ = 4;
        }
       
		channel = inputImg.channels();
		if (colorSpace == "hsv")
		{
			colorSpace = "gray";
		}

		if (channel == 1 || colorSpace == "rgb")
		{
			outputImg = inputImg;
			return outputImg;
		}
        outputImg = bici2::rgbConvertMex(inputImg, flag_, useSingle);
		return outputImg;
	}

	cv::Mat EdgeDetect::EdgeOrient(cv::Mat inputImg, int r)
	{
		cv::Mat E2, Dx, Dy, F;
		E2 = EdgeDetect::convTri(inputImg, r);
		std::vector<int> model = { -1, 2, -1 };
		std::vector<std::vector<int>> modelTranpose{ { -1 }, { 2 }, { -1 } };
		std::vector<std::vector<int>> bool_model{ { 1, 0, -1 }, { 0, 0, 0 }, { -1, 0, 1 } };
		
        cv::Mat array(3, 1, CV_8S);
        array.at<char>(0, 0) = -1;
        array.at<char>(1, 0) = -2;
        array.at<char>(2, 0) = -1;

        cv::Mat array2(3, 3, CV_8S);
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
                array2.at<char>(i, j) = bool_model[i][j];
        }

        cv::filter2D(E2, Dx, -1, model);
        cv::filter2D(E2, Dy, -1, array);
        cv::filter2D(E2, F, -1, array2);

        for (size_t i = 0; i < F.rows; i++) // F turn to bool value
        {
            for (size_t j = 0; j < F.cols; j++)
            if (F.at<float>(i, j) > 0)
            {
                F.at<float>(i, j) = 1;
            }
            else
            {
                F.at<float>(i, j) = 0;
            }
        }

        for (size_t i = 0; i < F.rows; i++) // Dy flip
        {
            for (size_t j = 0; j < F.cols; j++)
            if (F.at<float>(i, j) == 1)
            {
                Dy.at<float>(i, j) = -Dy.at<float>(i, j);
            }
        }

        //generate O
        cv::Mat O(F.rows, F.cols, CV_32FC1);
        for (size_t i = 0; i < F.rows; i++) // Dy flip
        {
            for (size_t j = 0; j < F.cols; j++)
                O.at<float>(i, j) = std::fmodf(std::atan2(Dy.at<float>(i, j), Dx.at<float>(i, j)), 3.1416);
        }

        return O;
	}

    cv::Mat EdgeDetect::GradientMag(const cv::Mat& img, int channels, int normRad, float normConst)
	{
		int full = 0;
		cv::Mat resultImg;
		resultImg = bici2::mGradMag(img,channels,full);
        cv::Mat temp[2];
        cv::split(resultImg, temp);
        cv::Mat M = temp[0];
        cv::Mat O = temp[1];
		if (normRad == 0) // normRad = 4;
		{
            return resultImg;
		}
		else
		{
			cv::Mat S = EdgeDetect::convTri(M, normRad);
            resultImg = bici2::mGradMagNorm(M, S, normConst);
            temp[0] = resultImg;
            cv::Mat result;
            cv::merge(temp, 2, result);
            return result;
		}
	}

    cv::Mat EdgeDetect::GradientHist(const cv::Mat& M, const cv::Mat& O, int binSize, int nOrients, int softbin)
    {
        return bici2::mGradHist(M, O, binSize, nOrients, softbin);
    }

	EdgeDetect::~EdgeDetect()
	{
	}

}