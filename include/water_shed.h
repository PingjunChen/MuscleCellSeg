// Pingjun Adapted Watershed from OpenCV

#ifndef MUSCLEMINER_WATER_SHED_H_
#define MUSCLEMINER_WATER_SHED_H_

#include <vector>

#include "opencv2/opencv.hpp"

#include "export.h"

namespace bici2
{
    // A node represents a pixel to label
    struct MUSCLEMINER_EXPORT WSNode
    {
        int next;
        int mask_ofs;
        int img_ofs;
    };

    // Queue for WSNodes
    struct MUSCLEMINER_EXPORT WSQueue
    {
        WSQueue() { first = last = 0; }
        int first, last;
    };

    MUSCLEMINER_EXPORT int allocWSNodes(std::vector<WSNode>& storage);
    MUSCLEMINER_EXPORT void WaterShedSeg(cv::Mat& _src, cv::Mat& _markers);
}


#endif // MUSCLEMINER_WATER_SHED_H_
