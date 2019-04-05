/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/

#ifndef MUSCLEMINER_WRAPPERS_H_
#define MUSCLEMINER_WRAPPERS_H_

#include <omp.h>
#include <typeinfo>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <cstdlib>

#include <string>

#include "opencv2/opencv.hpp"

#include "edge_model.h"
#include "export.h"

namespace bici2
{
    #define PI 3.14159265f
    typedef unsigned int uint32;
    typedef unsigned short uint16;
    typedef unsigned char uint8;

    // wrapper functions if compiling from C/C++
    MUSCLEMINER_EXPORT void wrError(const char *errormsg);
    MUSCLEMINER_EXPORT void* wrCalloc(size_t num, size_t size);
    MUSCLEMINER_EXPORT void* wrMalloc(size_t size);
    MUSCLEMINER_EXPORT void wrFree(void * ptr);

    // platform independent aligned memory allocation (see also alFree)
    MUSCLEMINER_EXPORT void* alMalloc(size_t size, int alignment);
    // platform independent alignned memory de-allocation (see also alMalloc)
    MUSCLEMINER_EXPORT void alFree(void* aligned);

    // imPad.h functions
    MUSCLEMINER_EXPORT void imPad(float *A, float *B, int h, int w, int d, int pt, int pb,
        int pl, int pr, int flag, float val);
    MUSCLEMINER_EXPORT cv::Mat ImPadMex(cv::Mat img, std::vector<int> pads, std::string pad_type);

    // sse.h functions - set, load and store values
    //MUSCLEMINER_EXPORT RETf SET(const float &x);
    //MUSCLEMINER_EXPORT RETf SET(float x, float y, float z, float w);
    //MUSCLEMINER_EXPORT RETi SET(const int &x);

    // set, load and store values
    __m128 SET(const float &x);
    __m128 SET(float x, float y, float z, float w);
    __m128i SET(const int &x);
    __m128 LD(const float &x);
    __m128 LDu(const float &x);
    __m128 STR(float &x, const __m128 y);
    __m128 STR1(float &x, const __m128 y); 
    __m128 STRu(float &x, const __m128 y);
    __m128 STR(float &x, const float y);

    // arithmetic operators
    __m128i ADD(const __m128i x, const __m128i y); 
    __m128 ADD(const __m128 x, const __m128 y);
    __m128 ADD(const __m128 x, const __m128 y, const __m128 z);
    __m128 ADD(const __m128 a, const __m128 b, const __m128 c, const __m128 &d);
    __m128 SUB(const __m128 x, const __m128 y); 
    __m128 MUL(const __m128 x, const __m128 y); 
    __m128 MUL(const __m128 x, const float y); 
    __m128 MUL(const float x, const __m128 y); 
    __m128 INC(__m128 &x, const __m128 y); 
    __m128 INC(float &x, const __m128 y); 
    __m128 DEC(__m128 &x, const __m128 y); 
    __m128 DEC(float &x, const __m128 y); 
    __m128 MIN1(const __m128 x, const __m128 y); 
    __m128 FMIN(const __m128 x, const __m128 y); 

    __m128 RCP(const __m128 x);
    __m128 RCPSQRT(const __m128 x);

    // logical operators
    __m128 AND(const __m128 x, const __m128 y);
    __m128i AND(const __m128i x, const __m128i y); 
    __m128 ANDNOT(const __m128 x, const __m128 y); 
    __m128 OR(const __m128 x, const __m128 y); 
    __m128 XOR(const __m128 x, const __m128 y); 

    // comparison operators
    __m128 CMPGT(const __m128 x, const __m128 y);
    __m128 CMPLT(const __m128 x, const __m128 y); 
    __m128i CMPGT(const __m128i x, const __m128i y);
    __m128i CMPLT(const __m128i x, const __m128i y);

    // conversion operators
    __m128 CVT(const __m128i x); 
    __m128i CVT(const __m128 x);

    // gradientMex.cpp functions
    MUSCLEMINER_EXPORT void grad1(float *I, float *Gx, float *Gy, int h, int w, int x);
    MUSCLEMINER_EXPORT void grad2(float *I, float *Gx, float *Gy, int h, int w, int d);
    MUSCLEMINER_EXPORT float* acosTable();
    MUSCLEMINER_EXPORT void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full);
    MUSCLEMINER_EXPORT void gradMagNorm(float *M, float *S, int h, int w, float norm);
    MUSCLEMINER_EXPORT void gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1,
        int nb, int n, float norm, int nOrients, bool full, bool interpolate);
    MUSCLEMINER_EXPORT void gradHist(float *M, float *O, float *H, int h, int w,
        int bin, int nOrients, int softBin, bool full);
    MUSCLEMINER_EXPORT float* hogNormMatrix(float *H, int nOrients, int hb, int wb, int bin);
    MUSCLEMINER_EXPORT void hogChannels(float *H, const float *R, const float *N,
        int hb, int wb, int nOrients, float clip, int type);
    MUSCLEMINER_EXPORT void hog(float *M, float *O, float *H, int h, int w, int binSize,
        int nOrients, int softBin, bool full, float clip);
    MUSCLEMINER_EXPORT void fhog(float *M, float *O, float *H, int h, int w, int binSize,
        int nOrients, int softBin, float clip);

    //#ifdef MATLAB_MEX_FILE
    MUSCLEMINER_EXPORT cv::Mat mGradMag(cv::Mat img, int channel, int full);
    MUSCLEMINER_EXPORT cv::Mat mGradMagNorm(cv::Mat M, cv::Mat S, float normConst);
    MUSCLEMINER_EXPORT cv::Mat mGradHist(cv::Mat img1, cv::Mat img2, int binSize, int nOrients, int softBin);

    //conConstMex.cpp functions
    MUSCLEMINER_EXPORT void convBoxY(float *I, float *O, int h, int r, int s);
    MUSCLEMINER_EXPORT void convBox(float *I, float *O, int h, int w, int d, int r, int s);
    MUSCLEMINER_EXPORT void conv11Y(float *I, float *O, int h, int side, int s);
    MUSCLEMINER_EXPORT void conv11(float *I, float *O, int h, int w, int d, int side, int s);
    MUSCLEMINER_EXPORT void convTriY(float *I, float *O, int h, int r, int s);
    MUSCLEMINER_EXPORT void convTri(float *I, float *O, int h, int w, int d, int r, int s);
    MUSCLEMINER_EXPORT void convTri1Y(float *I, float *O, int h, float p, int s);
    MUSCLEMINER_EXPORT void convTri1(float *I, float *O, int h, int w, int d, float p, int s);
    MUSCLEMINER_EXPORT void convMaxY(float *I, float *O, float *T, int h, int r);
    MUSCLEMINER_EXPORT void convMax(float *I, float *O, int h, int w, int d, int r);
    MUSCLEMINER_EXPORT cv::Mat convConstMex(std::string type, cv::Mat img, int r, int s);

    //rbgConvertMex.cpp functions
    MUSCLEMINER_EXPORT float* rgb2luv_setup(float z, float *mr, float *mg, float *mb,
        float &minu, float &minv, float &un, float &vn);
    MUSCLEMINER_EXPORT void rgb2luv(float *I, float *J, int n, float nrm);
    MUSCLEMINER_EXPORT void rgb2luv_sse(float *I, float *J, int n, float nrm);
    MUSCLEMINER_EXPORT void rgb2gray(float *I, float *J, int n, float nrm);
    MUSCLEMINER_EXPORT void normalize(float *I, float *J, int n, float nrm);
    MUSCLEMINER_EXPORT float* rgbConvert(float *I, int n, int d, int flag, float nrm);
    MUSCLEMINER_EXPORT cv::Mat rgbConvertMex(cv::Mat img, int flag, bool useSingle);

    //imResampleMex.cpp functions
    MUSCLEMINER_EXPORT void resampleCoef(int ha, int hb, int &n, int *&yas,
       int *&ybs, float *&wts, int bd[2], int pad);
   //void resample(double *A, double *B, int ha, int hb, int wa, int wb, int d, double r);
    MUSCLEMINER_EXPORT void resample(float *A, float *B, int ha, int hb, int wa, int wb, int d, float(r));
    MUSCLEMINER_EXPORT cv::Mat ImResampleMex(cv::Mat img, int height, int weight, int norm);

    //EdgeDetectMex.cpp
    //template<typename T> inline T min(T x, T y);
    MUSCLEMINER_EXPORT void buildLookupSs(uint32 *&cids1, uint32 *&cids2, int *dims, int w, int m);
    MUSCLEMINER_EXPORT uint32* buildLookup(int *dims, int w);
    MUSCLEMINER_EXPORT cv::Mat EdgeDetectMex(bici2::EdgeModel model, cv::Mat img, cv::Mat chns, cv::Mat chnsSs);

}
#endif // MUSCLEMINER_WRAPPERS_H_