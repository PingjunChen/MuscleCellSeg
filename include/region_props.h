/******************************************************************************
Copyright:  BICI2
Created:    18:5:2016 11:18
Filename:   region_props.h
Author:     Shiv

Purpose:    Calculate region properties
******************************************************************************/


#ifndef MUSCLEMINER_REGION_PROPS_H_
#define MUSCLEMINER_REGION_PROPS_H_

#include <vector>
#include "export.h"

namespace bici2
{
    enum MUSCLEMINER_EXPORT kRegProps
    {
        RP_BLANK_PROP = 0,
        RP_PIXEL_IDX_LIST = 1,
        RP_AREA = 2,
        RP_BOUNDING_BOX = 3,
        RP_SOLIDITY = 4,
    };

    struct MUSCLEMINER_EXPORT RegProps
    {
        public:
            float area;
            std::vector<float> boundingbox;
            float solidity;
            std::vector<int> pixelidxlist;
    };

}

#endif // MUSCLEMINER_REGION_PROPS_H_