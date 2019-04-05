/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   img_processing.cpp
Author:     Pingjun Chen

Purpose:    Implementations of common image processing functions.
******************************************************************************/


#include "img_processing.h"

namespace bici2
{
    void Display(const cv::Mat& img)
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


    cv::Mat LoadFromYML(std::string yml_path, std::string var_name)
    {
        cv::Mat load_img;
        cv::FileStorage yml_fs(yml_path, cv::FileStorage::READ);
        yml_fs[var_name] >> load_img;
        yml_fs.release();

        return load_img;
    }

    cv::Mat MorphClean(const cv::Mat& bin_img)
    {
        cv::Mat bin_clean = bin_img.clone();

        for (int y = 1; y < bin_img.rows-1; ++y)
        {
            for (int x = 1; x < bin_img.cols-1; ++x)
            {
                // If we already know it's not a maximum
                if (bin_img.at<uchar>(y, x) == 0)
                    continue;
                else
                {
                    if (bin_img.at<uchar>(y-1, x-1) == 0 && bin_img.at<uchar>(y-1, x) == 0
                        && bin_img.at<uchar>(y - 1, x + 1) == 0 && bin_img.at<uchar>(y, x-1) == 0
                        && bin_img.at<uchar>(y, x + 1) == 0 && bin_img.at<uchar>(y+1, x-1) == 0
                        && bin_img.at<uchar>(y + 1, x) == 0 && (bin_img.at<uchar>(y+1, x+1) == 0))
                            bin_clean.at<uchar>(y, x) = 0;
                }
            }
        }
        return bin_clean;
    }


    cv::Mat CreateConnectedComponent(const cv::Mat& ws_wt2)
    {
        int num_rows = ws_wt2.rows;
        int num_cols = ws_wt2.cols;

        cv::Mat bin_img = cv::Mat::zeros(num_rows, num_cols, CV_8U);
        for (int i_row = 0; i_row < num_rows; ++i_row)
        {
            for (int j_col = 0; j_col < num_cols; ++j_col)
            {
                // matlab ws_wt2 == 0
                bin_img.at<uchar>(i_row, j_col) =
                    ws_wt2.at<double>(i_row, j_col) < 1.0e-22 ? 255 : 0;
            }
        }
        //ws_wt2.convertTo(bin_img, CV_8U, 255.0);
        //cv::threshold(bin_img, bin_img, 1.0e-8, 255.0, CV_THRESH_BINARY_INV);

        // Apply connectedComponents to replace bwlabel in matlab
        cv::Mat labels;
        int num_cc = cv::connectedComponents(bin_img, labels, 8, CV_32S);

        //double min_val, max_val;
        //cv::minMaxLoc(labels, &min_val, &max_val);
        //std::cout << "min value of labels is " << min_val << std::endl;
        //std::cout << "max value of labels is " << max_val << std::endl;

        int half_rows = num_rows / 2;
        int half_cols = num_cols / 2;
        cv::Mat half_label = cv::Mat::zeros(half_rows, half_cols, CV_64F);
        for (int i_row = 0; i_row < half_rows; ++i_row)
        {
            for (int j_col = 0; j_col < half_cols; ++j_col)
            {
                half_label.at<double>(i_row, j_col) = labels.at<int>(i_row * 2 + 1, j_col * 2 + 1) - 1;
            }
        }

        ////double min_val, max_val;
        //cv::minMaxLoc(half_label, &min_val, &max_val);
        //std::cout << "min value of half_label is " << min_val << std::endl;
        //std::cout << "max value of half_label is " << max_val << std::endl;

        return half_label;
    }


    cv::Mat UCMMeanPB(const cv::Mat& ws_wt2, const cv::Mat& labels)
    {
        int fil = labels.rows;
        int col = labels.cols;

        // Assign mat to std::vector
        // Matlab is column first, transpose is needed.
        cv::Mat boundary = ws_wt2.clone().t();
        cv::Mat label = labels.clone().t();

        std::vector<double> local_boundaries;  // boundary
        if (boundary.isContinuous())
            local_boundaries.assign((double*)boundary.datastart, (double*)boundary.dataend);
        else
        {
            for (int i = 0; i < boundary.rows; ++i)
                local_boundaries.insert(local_boundaries.end(), (double*)boundary.ptr<uchar>(i),
                (double*)boundary.ptr<uchar>(i)+boundary.cols);
        }
        std::vector<double> partition;       // label
        if (label.isContinuous())
            partition.assign((double*)label.datastart, (double*)label.dataend);
        else
        {
            for (int i = 0; i < label.rows; ++i)
                partition.insert(partition.end(), (double*)label.ptr<uchar>(i),
                (double*)label.ptr<uchar>(i)+label.cols);
        }


        // assign partiton to int value
        int totcc = -1; // number of connected component
        std::vector<int> int_partition(fil*col);
        for (int px = 0; px < fil*col; ++px)
        {
            int_partition[px] = (int)partition[px];
            if (totcc < int_partition[px])
                totcc = int_partition[px];
        }
        if (totcc < 0)
        {
            std::cerr << "ERROR : number of connected components < 0 : \n";
        }
        totcc++;  //add one area for the background

        // apply compute_ucm
        std::vector<double> ucm((2 * fil + 1)*(2 * col + 1));
        compute_ucm(&local_boundaries[0], &int_partition[0], totcc, &ucm[0], fil, col);

        // super_ucm for storing the results
        cv::Mat super_ucm = cv::Mat(2 * col + 1, 2 * fil + 1, CV_64F);
        std::memcpy(super_ucm.data, &ucm[0], (2 * fil + 1) * (2 * col + 1) * sizeof(double));

        return super_ucm.t();
    }

    cv::Mat NormalizeImg(const cv::Mat& img, std::string fmt)
    {
        int num_rows = img.rows;
        int num_cols = img.cols;

        double beta[2] = { -2.7487, 11.1189 };
        double norm_para = 0.0602;

        cv::Mat norm_img = cv::Mat::zeros(num_rows, num_cols, CV_64F);
        for (int i_row = 0; i_row < num_rows; ++i_row)
        {
            for (int j_col = 0; j_col < num_cols; ++j_col)
            {
                double dotp = beta[0] + beta[1] * img.at<double>(i_row, j_col);
                double norm_p = ((1.0 / (1.0 + exp(-1.0*dotp))) - norm_para) / (1.0 - norm_para);
                norm_img.at<double>(i_row, j_col) = std::min(1.0, std::max(0.0, norm_p));
            }
        }

        // when strcmp(fmt,'imageSize')
        if (fmt == "imageSize")
        {
            int half_rows = (num_rows - 1) / 2;
            int half_cols = (num_cols - 1) / 2;
            cv::Mat half_norm = cv::Mat::zeros(half_rows, half_cols, CV_64F);
            for (int i_row = 0; i_row < half_rows; ++i_row)
            {
                for (int j_col = 0; j_col < half_cols; ++j_col)
                {
                    half_norm.at<double>(i_row, j_col) = norm_img.at<double>((i_row + 1) * 2, (j_col + 1) * 2);
                }
            }
            return half_norm;
        }
        else
        {
            return norm_img;
        }

        //return half_norm;
    }

    cv::Mat FindLocalMinima(const cv::Mat& image, int window_size)
    {
        cv::Mat regions = cv::Mat::ones(image.rows, image.cols, CV_8U);
        std::list<std::pair<int, int> > queue;

        for (int y = 0; y < image.rows; ++y)
        {
            for (int x = 0; x < image.cols; ++x)
            {
                // If we already know it's not a maximum
                if (!regions.at<uchar>(y, x))
                    continue;

                bool found = false;
                for (int yy = y - window_size; yy <= y + window_size; ++yy)
                {
                    for (int xx = x - window_size; xx <= x + window_size; ++xx)
                    {
                        if (yy < 0 || yy >= image.rows || xx < 0 || xx >= image.cols)
                            continue;
                        if ((yy == y) && (xx == x)) // do not consider with self
                            continue;

                        if (image.at<uchar>(yy, xx) < image.at<uchar>(y, x))
                        {
                            found = true;
                            break;
                        }
                    }
                }

                if (found)
                {
                    regions.at<uchar>(y, x) = 0;
                    queue.push_back(std::pair<int, int>(x, y));

                    while (!queue.empty())
                    {
                        int x2 = queue.front().first;
                        int y2 = queue.front().second;
                        queue.pop_front();

                        for (int yy2 = y2 - window_size; yy2 <= y2 + window_size; ++yy2)
                        {
                            for (int xx2 = x2 - window_size; xx2 <= x2 + window_size; ++xx2)
                            {
                                if (xx2 < 0 || yy2 < 0 || xx2 >= image.cols || yy2 >= image.rows)
                                    continue;
                                if ((yy2 == y2) && (xx2 == x2))
                                    continue;

                                if ((image.at<uchar>(yy2, xx2) == image.at<uchar>(y, x)) && regions.at<uchar>(yy2, xx2))
                                {
                                    regions.at<uchar>(yy2, xx2) = 0;
                                    queue.push_back(std::pair<int, int>(xx2, yy2));
                                }
                            }
                        }
                    } // while (!queue.empty())
                } // if (found)
            } // for (int x = 0;
        } // for (int y = 0;
        
        cv::Mat labels;
        int num_cc = cv::connectedComponents(regions, labels, 8, CV_32S);
        //std::cout << "Number of connected compoenent is " << num_cc << std::endl;
        return labels;
    }

    cv::Mat WatershedFull(const cv::Mat& image, const cv::Mat& marker)
    {
        // Apply watershed to those
        // cv::Mat imageu, markers, image3;
        //image.convertTo(imageu, CV_8U, 255);

        cv::Mat image3;
        cv::cvtColor(image, image3, cv::COLOR_GRAY2RGB);
        cv::Mat ws_res = marker.clone();

        // cv::watershed(image3, markers);
        WaterShedSeg(image3, ws_res);

        // OpenCV convention: -1 for boundaries, zone index start a 0
        // Matlab convention: 0 for boundaries, zone index start a 1
        ws_res += 1;
        

        return ws_res;
    }

    //cv::Mat ComputeWatershed(const cv::Mat& image, const cv::Mat& marker)
    //{
    //    cv::Mat regions = cv::Mat::ones(image.rows, image.cols, CV_32S);
    //    // std::priority_queue<float, std::vector<float>, std::greater<float> > queue;
    //    bici2::FifoPriorityQueue queue(FifoPriorityItemCompareFcn::LowestPriorityFirst);

    //    int pixel_num = image.rows * image.cols;
    //    std::vector<bool> s_pixel(pixel_num, false);
    //    //for (int ipixel = 0; ipixel < pixel_num; ++ipixel)
    //    //    s_pixel[ipixel] = false;

    //    // const int WSHED = 0;
    //    int col_len = image.cols;
    //    float* img_ptr = (float *)image.data;
    //    int* marker_ptr = (int *)marker.data;
    //    int* regions_ptr = (int *)regions.data;

    //    for (int ipixel = 0; ipixel < pixel_num; ++ipixel)
    //    {
    //        int irow = ipixel / image.cols; 
    //        int icol = ipixel % image.cols;
    //        // regions.at<int>(irow, icol) = marker.at<int>(irow, icol);
    //        *(regions_ptr + irow*col_len + icol) = *(marker_ptr + irow*col_len + icol);
    //        //if (marker.at<int>(irow, icol) != 0)
    //        if (*(marker_ptr + irow*col_len + icol))
    //        {
    //            s_pixel[ipixel] = true;
    //            for (int yy = irow - 1; yy <= irow + 1; ++yy)
    //            {
    //                for (int xx = icol - 1; xx <= icol + 1; ++xx)
    //                {
    //                    if (((yy == irow) && (xx == icol)) 
    //                        || (xx < 0 || yy < 0 || xx >= image.cols || yy >= image.rows))
    //                        continue;

    //                    int q_ind = yy * image.cols + xx;
    //                    // if (!s_pixel[q_ind] && (marker.at<int>(yy, xx) == 0))
    //                    if (!s_pixel[q_ind] && (*(marker_ptr + yy*col_len + xx) == 0))
    //                    {
    //                        s_pixel[q_ind] = true;
    //                        // queue.push(q_ind, image.at<float>(yy, xx));
    //                        queue.push(q_ind, *(img_ptr + yy*col_len +xx));
    //                    }
    //                }
    //            }
    //        }
    //    }

    //    // int loop_times = 0;
    //    // recursively fill the basins
    //    while (!queue.isEmpty())
    //    {
    //        // loop_times++;
    //        int p_ind = queue.topData();
    //        float p_val = queue.topPriority();
    //        queue.pop();
    //
    //        int label = 0;
    //        bool watershed = false;

    //        int irow = p_ind / image.cols;
    //        int icol = p_ind % image.cols;
    //        for (int yy = irow - 1; yy <= irow + 1; ++yy)
    //        {
    //            for (int xx = icol - 1; xx <= icol + 1; ++xx)
    //            {
    //                // ??? border condition, may slow the speed.
    //                if (((yy == irow) && (xx == icol))
    //                    || (xx < 0 || yy < 0 || xx >= image.cols || yy >= image.rows))
    //                    continue;

    //                if ((*(regions_ptr + yy*col_len + xx) != 0) && !watershed)
    //                {
    //                    if ((label != 0) && (*(regions_ptr + yy*col_len + xx) != label))
    //                        watershed = true;
    //                    else
    //                        label = *(regions_ptr + yy * col_len + xx);
    //                }
    //            }
    //        }

    //        if (!watershed)
    //        {
    //            // *(regions_ptr+irow*col_len+icol) = label;
    //            *(regions_ptr + p_ind) = label;
    //            for (int yy = irow - 1; yy <= irow + 1; ++yy)
    //            {
    //                for (int xx = icol - 1; xx <= icol + 1; ++xx)
    //                {
    //                    if (((yy == irow) && (xx == icol))
    //                        || (xx < 0 || yy < 0 || xx >= image.cols || yy >= image.rows))
    //                        continue;
    //                    
    //                    int q_ind = yy * image.cols + xx;
    //                    if (!s_pixel[q_ind])
    //                    {
    //                        s_pixel[q_ind] = true;
    //                        queue.push(q_ind, std::max(*(img_ptr + yy*col_len + xx), p_val));

    //                    }
    //                }
    //            }
    //        }
    //    }
    //    return regions;
    //}

    //  works similar to RegionProps as MATLAB, currently only for 4 properties
    void RegionProps(const cv::Mat& in, std::vector<RegProps> &out,
        kRegProps kregProp1, kRegProps kregProp2, kRegProps kregProp3, kRegProps kregProp4)
    {
        //  Find connected components with stats
        cv::Mat in_conn, stats, centroids;
        cv::connectedComponentsWithStats(in, in_conn, stats, centroids);

        for (int i = 1; i<stats.rows; ++i) {
            RegProps temp;

            //  Find area, complete
            if (kregProp1 == RP_AREA || kregProp2 == RP_AREA || kregProp3 == RP_AREA || kregProp4 == RP_AREA) {
                temp.area = stats.at<int>(i, 4);
            }

            //  Find bounding box, complete
            if (kregProp1 == RP_BOUNDING_BOX || kregProp2 == RP_BOUNDING_BOX || kregProp3 == RP_BOUNDING_BOX || kregProp4 == RP_BOUNDING_BOX) {
                temp.boundingbox.push_back(stats.at<int>(i, 0) + 0.5);
                temp.boundingbox.push_back(stats.at<int>(i, 1) + 0.5);
                temp.boundingbox.push_back(stats.at<int>(i, 2));
                temp.boundingbox.push_back(stats.at<int>(i, 3));
            }

            //  find solidity, have to find covex hull area, not working
            if (kregProp1 == RP_SOLIDITY || kregProp2 == RP_SOLIDITY || kregProp3 == RP_SOLIDITY || kregProp4 == RP_SOLIDITY) {
                //  isolate the blob
                /*	cv::Mat blob = cv::Mat::zeros(in_conn.rows, in_conn.cols, CV_8U);
                blob.setTo(1, in_conn == i);

                //  find contours
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(blob, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

                //  find hull contour
                std::vector<cv::Point> hullcontour;
                cv::convexHull(contours[0], hullcontour);

                temp.solidity = float(stats.at<int>(i, 4)) / cv::contourArea(hullcontour); */
            }

            //  pixel ID list, complete
            if (kregProp1 == RP_PIXEL_IDX_LIST || kregProp2 == RP_PIXEL_IDX_LIST || kregProp3 == RP_PIXEL_IDX_LIST || kregProp4 == RP_PIXEL_IDX_LIST) {
                int* in_conn_ptr = (int *)(in_conn.data);
                int size = 0;
                for (int j = 0; j < in_conn.rows; ++j) {
                    for (int k = 0; k < in_conn.cols; ++k) {
                        if (in_conn_ptr[j*in_conn.cols + k] == i) {
                            //temp.pixelidxlist.push_back(k*in_conn.rows + j + 1);
                            temp.pixelidxlist.push_back(j*in_conn.rows + k);
                            size++;
                        }
                    }
                    if (size == stats.at<int>(i, 4)) {
                        break;
                    }
                }
            }

            std::sort(temp.pixelidxlist.begin(), temp.pixelidxlist.end());

            out.push_back(temp);
        }
    }

}