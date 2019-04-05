/******************************************************************************
Copyright:  BICI2
Created:    18:6:2016 11:18
Filename:   edge_model.h
Author:     Pingjun Chen

Purpose:    Load structured random forest model
******************************************************************************/

#ifndef MUSCLEMINER_EDGEMODEL_H_
#define MUSCLEMINER_EDGEMODEL_H_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <limits>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "export.h"

namespace bici2
{
    struct MUSCLEMINER_EXPORT EdgeModelOpts
    {
    public:
        EdgeModelOpts() {}
        ~EdgeModelOpts() {}
    public:
        int imWidth;
        int gtWidth;
        int nEdgeBins;
        int nPos;
        int nNeg;
        int nImgs;
        int nTrees;
        float fracFtrs;
        int minCount;
        int minChild;
        int maxDepth;
        std::string discretize;
        int nSamples;
        int nClasses;
        std::string split;
        int nOrients;
        float grdSmooth;
        float chnSmooth;
        int simSmooth;
        int normRad;
        int shrink;
        int nCells;
        int rgbd;
        int stride;
        int multiscale;
        int nTreesEval;
        int nThreads;
        int nms;
        int seed;
        int useParfor;
        std::string modelDir;
        std::string modelFnm;
        std::string bsdsDir;
        int nChns;
        int nChnFtrs;
        int nSimFtrs;
        int nTotFtrs;
    };

    class MUSCLEMINER_EXPORT EdgeModel
    {
    public:
        EdgeModel() {};
        ~EdgeModel() {};
        void ParseOpts(const std::string& path);
        void ParseThrs(const std::string& path);
        void ParseFids(const std::string& path);
        void ParseChild(const std::string& path);
        void ParseCount(const std::string& path);
        void ParseDepth(const std::string& path);
        void ParseEbins(const std::string& path);
        void ParseEbnds(const std::string& path);

        void ParseEdgeModel(const std::string& opts_path,
            const std::string& thrs_path,
            const std::string& fids_path,
            const std::string& child_path,
            const std::string& count_path,
            const std::string& depth_path,
            const std::string& ebins_path,
            const std::string& ebnds_path);

    public:
        struct EdgeModelOpts opts;
        std::vector<float> thrs;
        std::vector<unsigned int> fids;
        std::vector<unsigned int> child;
        std::vector<unsigned int> count;
        std::vector<unsigned int> depth;
        std::vector<unsigned short> eBins;
        std::vector<unsigned int> eBnds;
    };
}

#endif // MUSCLEMINER_EDGEMODEL_H_