/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   frangi.cpp
Author:     Shiv

Purpose:    Frangi implementation
******************************************************************************/


#include "frangi.h"

namespace bici2
{
	Frangi::Frangi()
	{
		// std::cout << "Hello CoinCount" << std::endl;
	}

	Frangi::Frangi(std::string path)
	{
		this->imgpath_ = path;
		this->img_ = cv::imread(imgpath_);
		//cv::FileStorage yml_fs(path, cv::FileStorage::READ);
		//yml_fs["edgeMap"] >> this->img_;
		//yml_fs.release();
		cv::namedWindow("Display");
		imshow("Display", img_);
		cv::waitKey(0);
	}

	void Frangi::SetPath(std::string path)
	{
		this->imgpath_ = path;
		this->img_ = cv::imread(imgpath_);
	}

    void Frangi::SetMat(const cv::Mat& img)
    {
        this->img_ = img.clone();
    }

	cv::Mat Frangi::GetMat() const
	{
		return this->img_;
	}

	//  Help taken from https://github.com/ntnu-bioopt/libfrangi

	//  Hessian2D function for post processing
	//  d_xx, dxy, dyy: output filtered images
	//  img: input image
	//  sigma:  threshold for filtering
	//  all comments with '%' are MATLAB code
	//  COMPLETED (for simple test cases)
	//  This function is a little different from
	//  the MATLAB implementation because cv::filter2D
	//  does not zero-pad the borders, instead
	//  does the same as 'replicate' option in
	//  imfilter in MATLAB
	//  should not be a problem, as only borders
	//  would be affected, if needed, zero-padding
	//  can be implemented
	void Frangi::Hessian2D(
			cv::Mat& d_xx,  //  d_xx, d_xy, d_yy memory will be allocated by filter2D function
			cv::Mat& d_xy,
			cv::Mat& d_yy,
			const cv::Mat& img,
			const float sigma
			)
	{
		//  % Make kernel coordinates
		//  %  Build the Gaussian 2nd derivative filters
		//  implement:
		//  [X,Y]=ndgrid(-3*round(sigma):3*round(sigma));
		//  DGaussxx = 1 / (2 * pi*Sigma ^ 4) * (X. ^ 2 / Sigma ^ 2 - 1).*exp(-(X. ^ 2 + Y. ^ 2) / (2 * Sigma ^ 2));
		//  DGaussxy = 1 / (2 * pi*Sigma ^ 6) * (X.*Y).*exp(-(X. ^ 2 + Y. ^ 2) / (2 * Sigma ^ 2));
		//  DGaussyy = DGaussxx';
		//  COMPLETED

		// dgaussxx, dgaussxy,dgaussyy are same dimensions as x and y and of type Mat
		// create these vector 2d matrices
		int kernel_size = 2*int(round(3*sigma)) + 1;
		cv::Mat dgaussxx = cv::Mat(kernel_size, kernel_size, CV_32F),
			dgaussxy = cv::Mat(kernel_size, kernel_size, CV_32F),
			dgaussyy = cv::Mat(kernel_size, kernel_size, CV_32F);


		//  optimization
		//  1 / (2 * pi*Sigma ^ 4) = k1xx
		//  Sigma ^ 2 = k2xx
		//  (2 * Sigma ^ 2) = k3
		//  1 / (2 * pi*Sigma ^ 6) = k1xy
		//  (2 * Sigma ^ 2) = k3
		// t3 is internal variable for optimization
        float k1xx = 1.0f / (2.0f * CV_PI * pow(sigma, 4));  //k is used for global constants, change names
		float k2xx = pow(sigma, 2);
		float k3 = 2.0f * k2xx;
        float k1xy = 1.0f / (2.0f * CV_PI * pow(sigma, 6));
		float t3;  // temporary variable, t1,t2 declared earlier as vectors

		//  performing the calculations
		//  can be optimized further
		//  replace with safe iterator method
		//  http://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html
		int start_x_y = -int(round(3 * sigma)), end_x_y = int(round(3 * sigma));
		for (int x = start_x_y, i=0; x <= end_x_y; ++x, ++i) {
			for (int y = start_x_y, j=0; y <= end_x_y; ++y, ++j) {
				t3 = exp(-(x*x + y*y) / k3);
				dgaussxx.at<float>(i,j) = k1xx * (((x*x) / k2xx) - 1) * t3;
				dgaussxy.at<float>(i, j) = k1xy * (x * y) * t3;
				dgaussyy.at<float>(j, i) = dgaussxx.at<float>(i, j);
			}
		}

		//  %  implement the filter
		//  implement:
		//  Dxx = imfilter(I, DGaussxx, 'conv');
		//  Dxy = imfilter(I, DGaussxy, 'conv');
		//  Dyy = imfilter(I, DGaussyy, 'conv');
		//  design dedicated filter for 'conv'
		//  does opencv have an implementation? yes!
		//  http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html
		//  COMPLETED

		//  prepare the filter
		cv::Point anchor(kernel_size - kernel_size / 2 - 1, kernel_size - kernel_size / 2 - 1);  //  anchor is (0,0) in MATLAB
																								 //  shifing anchor for convolutional filtering
		double delta = 0;
		cv::Mat flipped_dgaussxx, flipped_dgaussxy, flipped_dgaussyy;  //  flipped kernels for convolutional filtering
		cv::flip(dgaussxx, flipped_dgaussxx, -1);
		cv::flip(dgaussxy, flipped_dgaussxy, -1);
		cv::flip(dgaussyy, flipped_dgaussyy, -1);

		//  applying the filter
		cv::filter2D(img, d_xx, -1, flipped_dgaussxx, anchor);
		cv::filter2D(img, d_xy, -1, flipped_dgaussxy, anchor);
		cv::filter2D(img, d_yy, -1, flipped_dgaussyy, anchor);

	}

	//  lambda2, lambda1, Ix, Iy are cv::Mat type and their memory is allocated
	//  inside the function, so no need to preallocate
	//  They are passed by reference
	//  TODO: Perform speed comparisons between various implementations
	void Frangi::Eig2image(cv::Mat& lambda1,
		cv::Mat& lambda2,
		cv::Mat& Ix,
		cv::Mat& Iy,
		const cv::Mat& d_xx,
		const cv::Mat& d_xy,
		const cv::Mat& d_yy
		)
	{
		//  METHOD FROM https://github.com/ntnu-bioopt/libfrangi/blob/master/src/frangi.cpp

		cv::Mat tmp, tmp2;
		tmp2 = d_xx - d_yy;
		cv::sqrt(tmp2.mul(tmp2) + (4.0f * d_xy.mul(d_xy)), tmp);
		cv::Mat v2x = 2 * d_xy;
		cv::Mat v2y = d_yy - d_xx + tmp;

		//normalize
		cv::Mat mag;
		cv::sqrt((v2x.mul(v2x) + v2y.mul(v2y)), mag);
		cv::Mat v2xtmp = v2x.mul(1.0f / mag);
		v2xtmp.copyTo(v2x, mag != 0);
		cv::Mat v2ytmp = v2y.mul(1.0f / mag);
		v2ytmp.copyTo(v2y, mag != 0);

		//eigenvectors are orthogonal
		cv::Mat v1x, v1y;
		v2y.copyTo(v1x);
		v1x = -1 * v1x;
		v2x.copyTo(v1y);

		//compute eigenvalues
		cv::Mat mu1 = 0.5*(d_xx + d_yy + tmp);
		cv::Mat mu2 = 0.5*(d_xx + d_yy - tmp);

		//sort eigenvalues by absolute value abs(Lambda1) < abs(Lamda2)
		cv::Mat check = abs(mu1) > abs(mu2);
		mu1.copyTo(lambda1); mu2.copyTo(lambda1, check);
		mu2.copyTo(lambda2); mu1.copyTo(lambda2, check);

		v1x.copyTo(Ix); v2x.copyTo(Ix, check);
		v1y.copyTo(Iy); v2y.copyTo(Iy, check);
	}

	void Frangi::FrangiFilter2D(cv::Mat& out_img, const cv::Mat& img)
	{
		//  dummy output variables
		cv::Mat whatscale, outangles;

		//  options that are being harcoded here
		//  check if hard coding is needed
		std::vector< float > sigmas = { 1, 2, 3, 4, 5 };
		float FrangiScaleRatio = 1;
		float FrangiBetaOne = 1;
		float FrangiBetaTwo = 10;

		//  main code
		//  implement:
		//  beta = 2 * options.FrangiBetaOne ^ 2;
		//  c = 2 * options.FrangiBetaTwo ^ 2;
		float beta = 2.0f * FrangiBetaOne * FrangiBetaOne;
		float c = 2.0f * FrangiBetaTwo * FrangiBetaTwo;

		//  % Make matrices to store all filterd images
		//	ALLfiltered = zeros([size(I) length(sigmas)]);
		//  ALLangles = zeros([size(I) length(sigmas)]);
		std::vector<cv::Mat> all_filtered;
		std::vector<cv::Mat> all_angles;

		for (int i = 0; i < sigmas.size(); i++)
		{
			//  % Make 2D hessian
			//  [Dxx, Dxy, Dyy] = Hessian2D(I, sigmas(i));
			cv::Mat d_xx, d_xy, d_yy;
			Hessian2D(d_xx, d_xy, d_yy, img, sigmas[i]);

			//  % Correct for scale
			//  Dxx = (sigmas(i) ^ 2)*Dxx;
			d_xx = d_xx*sigmas[i]*sigmas[i];
			d_xy = d_xy*sigmas[i]*sigmas[i];
			d_yy = d_yy*sigmas[i]*sigmas[i];

			//  % Calculate(abs sorted) eigenvalues and vectors
			//  [Lambda2, Lambda1, Ix, Iy] = eig2image(Dxx, Dxy, Dyy);
			cv::Mat lambda1, lambda2, Ix, Iy;
			Eig2image(lambda1, lambda2, Ix, Iy, d_xx, d_xy, d_yy);

			//  % Compute the direction of the minor eigenvector
			//  angles = atan2(Ix, Iy);
			cv::Mat angles;
			cv::phase(Ix, Iy, angles);  //  compiles without cv::, check difference in output
			all_angles.push_back(angles);

			//  % Compute some similarity measures
			//	Lambda1(Lambda1 == 0) = eps; %eps = nextafterf(0, 1) in C++
			//  Rb = (Lambda2. / Lambda1). ^ 2;
			//  S2 = Lambda1. ^ 2 + Lambda2. ^ 2;
			lambda2.setTo(nextafterf(0, 1), lambda2 == 0);
			cv::Mat Rb = lambda1.mul(1.0f / lambda2);
			Rb = Rb.mul(Rb);
			cv::Mat S2 = lambda1.mul(lambda1) + lambda2.mul(lambda2);

			//  % Compute the output image
			// 	Ifiltered = exp(-Rb / beta) .*(ones(size(I)) - exp(-S2 / c));
			cv::Mat tmp1, tmp2;
			exp(-Rb / beta, tmp1);
			exp(-S2 / c, tmp2);

			cv::Mat img_filtered = tmp1.mul(cv::Mat::ones(img.rows, img.cols, img.type()) - tmp2);

			img_filtered.setTo(0, lambda2 > 0); //  options.BlackWhite = false (hard coded)

			//  store results
			all_filtered.push_back(img_filtered);
		}

		all_filtered[0].copyTo(out_img);
		all_filtered[0].copyTo(whatscale);
		all_filtered[0].copyTo(outangles);
		whatscale.setTo(sigmas[0]);

		//find element-wise maximum across all accumulated filter results
		for (int i = 1; i < all_filtered.size(); i++)
		{
			out_img = cv::max(out_img, all_filtered[i]);
			whatscale.setTo(sigmas[i], all_filtered[i] == out_img);
			all_angles[i].copyTo(outangles, all_filtered[i] == out_img);
		}

	}

    cv::Mat Frangi::Anisodiff2D(cv::Mat& img)
    {
        //  hard coding options
        int num_iter = 15;
        float delta_t = 1.0f / 7.0f;
        int kappa = 30;
        //  int option = 2; useless if hardcoded

        //  should the image be converted to double?

        //  PDE (partial differential equation) initial condition.
        cv::Mat diff_img;
        img.copyTo(diff_img);

        //  Center pixel distances.
        float dx = 1.0f;
        float dy = 1.0f;
        float dd = sqrt(2.0f);

        //  2D convolution masks - finite differences. unflipped
        cv::Mat uf_hN = (cv::Mat_<float>(3, 3) << 0, 1, 0, 0, -1, 0, 0, 0, 0);
        cv::Mat uf_hS = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, -1, 0, 0, 1, 0);
        cv::Mat uf_hE = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, -1, 1, 0, 0, 0);
        cv::Mat uf_hW = (cv::Mat_<float>(3, 3) << 0, 0, 0, 1, -1, 0, 0, 0, 0);
        cv::Mat uf_hNE = (cv::Mat_<float>(3, 3) << 0, 0, 1, 0, -1, 0, 0, 0, 0);
        cv::Mat uf_hSE = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, -1, 0, 0, 0, 1);
        cv::Mat uf_hSW = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, -1, 0, 1, 0, 0);
        cv::Mat uf_hNW = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, 0);

        //  flip these kernels
        cv::Mat hN, hS, hE, hW, hNE, hSE, hSW, hNW;
        cv::flip(uf_hN, hN, -1);
        cv::flip(uf_hS, hS, -1);
        cv::flip(uf_hE, hE, -1);
        cv::flip(uf_hW, hW, -1);
        cv::flip(uf_hNE, hNE, -1);
        cv::flip(uf_hSE, hSE, -1);
        cv::flip(uf_hSW, hSW, -1);
        cv::flip(uf_hNW, hNW, -1);

        //   preallocate the matrices
        cv::Mat nablaN = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat nablaS = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat nablaW = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat nablaE = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat nablaNE = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat nablaSE = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat nablaSW = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat nablaNW = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);

        cv::Mat cN = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat cS = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat cW = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat cE = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat cNE = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat cSE = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat cSW = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);
        cv::Mat cNW = cv::Mat(diff_img.rows, diff_img.cols, CV_32F);

        //  prepare the filter anchor
        cv::Point anchor(hN.cols - hN.cols / 2 - 1, hN.rows - hN.rows / 2 - 1);  //  anchor is same for all

        //  Anisotropic diffusion.
        for (int t = 0; t < num_iter; ++t) {
            //  Finite differences.
            cv::filter2D(diff_img, nablaN, -1, hN, anchor);
            cv::filter2D(diff_img, nablaS, -1, hS, anchor);
            cv::filter2D(diff_img, nablaW, -1, hW, anchor);
            cv::filter2D(diff_img, nablaE, -1, hE, anchor);
            cv::filter2D(diff_img, nablaNE, -1, hNE, anchor);
            cv::filter2D(diff_img, nablaSE, -1, hSE, anchor);
            cv::filter2D(diff_img, nablaSW, -1, hSW, anchor);
            cv::filter2D(diff_img, nablaNW, -1, hNW, anchor);

            //  Diffusion function (assuming option = 2) 
            cN = nablaN*(1.0f / kappa);
            cN = cN.mul(cN);
            cN = 1 + cN;
            cv::pow(cN, -1.0f, cN);

            cS = nablaS*(1.0f / kappa);
            cS = cS.mul(cS);
            cS = 1 + cS;
            cv::pow(cS, -1.0f, cS);

            cW = nablaW*(1.0f / kappa);
            cW = cW.mul(cW);
            cW = 1 + cW;
            cv::pow(cW, -1.0f, cW);

            cE = nablaE*(1.0f / kappa);
            cE = cE.mul(cE);
            cE = 1 + cE;
            cv::pow(cE, -1.0f, cE);

            cNE = nablaNE*(1.0f / kappa);
            cNE = cNE.mul(cNE);
            cNE = 1 + cNE;
            cv::pow(cNE, -1.0f, cNE);

            cSE = nablaSE*(1.0f / kappa);
            cSE = cSE.mul(cSE);
            cSE = 1 + cSE;
            cv::pow(cSE, -1.0f, cSE);

            cSW = nablaSW*(1.0f / kappa);
            cSW = cSW.mul(cSW);
            cSW = 1 + cSW;
            cv::pow(cSW, -1.0f, cSW);

            cNW = nablaNW*(1.0f / kappa);
            cNW = cNW.mul(cNW);
            cNW = 1 + cNW;
            cv::pow(cNW, -1.0f, cNW);

            //  Discrete PDE solution
            diff_img = diff_img +
                delta_t*(
                (1.0f / (dy * dy))*cN.mul(nablaN) + (1.0f / (dy * dy))*cS.mul(nablaS) +
                (1.0f / (dx * dx))*cW.mul(nablaW) + (1.0f / (dx * dx))*cE.mul(nablaE) +
                (1.0f / (dd * dd))*cNE.mul(nablaNE) + (1.0f / (dd * dd))*cSE.mul(nablaSE) +
                (1.0f / (dd * dd))*cSW.mul(nablaSW) + (1.0f / (dd * dd))*cNW.mul(nablaNW));
        }

        return diff_img;
    }


    //  Using connected components instead of contours
    void Frangi::RemoveSmallBlobs(cv::Mat& im, double size)
    {
        // Only accept CV_8UC1
        if (im.channels() != 1 || im.type() != CV_8U)
            return;

        //  find connected components
        cv::Mat im_conn;
        cv::connectedComponents(im, im_conn);

        //  find maximum value to find size of table
        double minVal, maxVal;
        cv::minMaxLoc(im_conn, &minVal, &maxVal);
        int table_size = (int)(maxVal);
        table_size++;

        //  allocate memory to table
        int* table = new int[table_size];

        //  initialize table to 0
        for (int i = 0; i < table_size; ++i)
            table[i] = 0;

        //  pointer to connected components image
        int* im_conn_ptr = (int *)(im_conn.data);

        //  use sum table to find the number of each value in image
        for (int i = 0; i < im.rows; ++i) {
            for (int j = 0; j < im.cols; ++j) {
                table[im_conn_ptr[i*im.cols + j]]++;
            }
        }

        //  pointer to input image
        uchar* im_ptr = (uchar *)(im.data);

        //  set all blobs of size less than 'size' to 0
        for (int i = 0; i < im.rows; ++i) {
            for (int j = 0; j < im.cols; ++j) {
                im_ptr[i*im.cols + j] = (im_ptr[i*im.cols + j] != 0) && (table[(im_conn_ptr[i*im.cols + j])]>size);
            }
        }

        //  deallcoate memory
        delete[] table;
    }


    cv::Mat Frangi::ApplyFrangi()
    {
        int post = 1;

        cv::Mat ridge;
        FrangiFilter2D(ridge, this->img_);

        cv::Mat output_ridge;
        if (post == 1) {
            cv::Mat ridge_8bit, bw_ridge, ridge_scaled;
            ridge_scaled = ridge * 255;
            ridge_scaled.convertTo(ridge_8bit, CV_8U);
            cv::threshold(ridge_8bit, bw_ridge, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
            RemoveSmallBlobs(bw_ridge, 5000);
            ridge.copyTo(output_ridge, bw_ridge != 0);
        }
        else
            ridge.copyTo(output_ridge);

        cv::Mat edge_map;
        edge_map = Anisodiff2D(output_ridge);

        return edge_map;
	}

	Frangi::~Frangi()
	{
		// std::cout << "Byebye CoinCount" << std::endl;
	}
}