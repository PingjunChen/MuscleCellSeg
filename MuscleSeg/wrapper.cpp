/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/

#include "wrapper.h"

namespace bici2
{
    void wrError(const char *errormsg)
    {
        throw errormsg;
    }
    void* wrCalloc(size_t num, size_t size)
    {
        return calloc(num, size);
    }
    void* wrMalloc(size_t size)
    {
        return malloc(size);
    }
    void wrFree(void * ptr)
    {
        free(ptr);
    }
    // memory allocation
    void* alMalloc(size_t size, int alignment)
    {
        const size_t pSize = sizeof(void*), a = alignment - 1;
        void *raw = wrMalloc(size + a + pSize);
        void *aligned = (void*)(((size_t)raw + pSize + a) & ~a);
        *(void**)((size_t)aligned - pSize) = raw;
        return aligned;
    }
    // memory de-allocation
    void alFree(void* aligned)
    {
        void* raw = *(void**)((char*)aligned - sizeof(void*));
        wrFree(raw);
    }

// imPad functions
    void imPad(float *A, float *B, int h, int w, int d, int pt, int pb,
        int pl, int pr, int flag, float val)
    {
        int h1 = h + pt, hb = h1 + pb, w1 = w + pl, wb = w1 + pr, x, y, z, mPad;
        int ct = 0, cb = 0, cl = 0, cr = 0;
        if (pt<0) { ct = -pt; pt = 0; } if (pb<0) { h1 += pb; cb = -pb; pb = 0; }
        if (pl<0) { cl = -pl; pl = 0; } if (pr<0) { w1 += pr; cr = -pr; pr = 0; }
        int *xs, *ys; x = pr>pl ? pr : pl; y = pt>pb ? pt : pb; mPad = x>y ? x : y;
        bool useLookup = ((flag == 2 || flag == 3) && (mPad>h || mPad > w))
            || (flag == 3 && (ct || cb || cl || cr));
        // helper macro for padding
#define PAD(XL,XM,XR,YT,YM,YB) \
        for (x = 0; x < pl; x++) for (y = 0; y < pt; y++) B[x*hb + y] = A[(XL + cl)*h + YT + ct]; \
        for (x = 0; x < pl; x++) for (y = pt; y < h1; y++) B[x*hb + y] = A[(XL + cl)*h + YM + ct]; \
        for (x = 0; x < pl; x++) for (y = h1; y < hb; y++) B[x*hb + y] = A[(XL + cl)*h + YB - cb]; \
        for (x = pl; x < w1; x++) for (y = 0; y < pt; y++) B[x*hb + y] = A[(XM + cl)*h + YT + ct]; \
        for (x = pl; x < w1; x++) for (y = h1; y < hb; y++) B[x*hb + y] = A[(XM + cl)*h + YB - cb]; \
        for (x = w1; x < wb; x++) for (y = 0; y < pt; y++) B[x*hb + y] = A[(XR - cr)*h + YT + ct]; \
        for (x = w1; x < wb; x++) for (y = pt; y < h1; y++) B[x*hb + y] = A[(XR - cr)*h + YM + ct]; \
        for (x = w1; x < wb; x++) for (y = h1; y < hb; y++) B[x*hb + y] = A[(XR - cr)*h + YB - cb];
        // build lookup table for xs and ys if necessary
        if (useLookup) {
            xs = (int*)wrMalloc(wb*sizeof(int)); int h2 = (pt + 1) * 2 * h;
            ys = (int*)wrMalloc(hb*sizeof(int)); int w2 = (pl + 1) * 2 * w;
            if (flag == 2) {
                for (x = 0; x < wb; x++) { z = (x - pl + w2) % (w * 2); xs[x] = z < w ? z : w * 2 - z - 1; }
                for (y = 0; y < hb; y++) { z = (y - pt + h2) % (h * 2); ys[y] = z < h ? z : h * 2 - z - 1; }
            }
            else if (flag == 3) {
                for (x = 0; x < wb; x++) xs[x] = (x - pl + w2) % w;
                for (y = 0; y < hb; y++) ys[y] = (y - pt + h2) % h;
            }
        }
        // pad by appropriate value
        for (z = 0; z < d; z++) {
            // copy over A to relevant region in B
            for (x = 0; x < w - cr - cl; x++)
                memcpy(B + (x + pl)*hb + pt, A + (x + cl)*h + ct, sizeof(float)*(h - ct - cb));
            // set boundaries of B to appropriate values
            if (flag == 0 && val != 0) { // "constant"
                for (x = 0; x < pl; x++) for (y = 0; y < hb; y++) B[x*hb + y] = val;
                for (x = pl; x < w1; x++) for (y = 0; y < pt; y++) B[x*hb + y] = val;
                for (x = pl; x < w1; x++) for (y = h1; y < hb; y++) B[x*hb + y] = val;
                for (x = w1; x < wb; x++) for (y = 0; y < hb; y++) B[x*hb + y] = val;
            }
            else if (useLookup) { // "lookup"
                PAD(xs[x], xs[x], xs[x], ys[y], ys[y], ys[y]);
            }
            else if (flag == 1) {  // "replicate"
                PAD(0, x - pl, w - 1, 0, y - pt, h - 1);
            }
            else if (flag == 2) { // "symmetric"
                PAD(pl - x - 1, x - pl, w + w1 - 1 - x, pt - y - 1, y - pt, h + h1 - 1 - y);
            }
            else if (flag == 3) { // "circular"
                PAD(x - pl + w, x - pl, x - pl - w, y - pt + h, y - pt, y - pt - h);
            }
            A += h*w;  B += hb*wb;
        }
        if (useLookup) { wrFree(xs); wrFree(ys); }
#undef PAD
    }
    cv::Mat ImPadMex(cv::Mat img, std::vector<int> pads, std::string pad_type)
    {
        // processing dimensions
        int ndims = img.dims;
        std::vector<int> ns;
        ns.push_back(img.rows);
        ns.push_back(img.cols);
        ns.push_back(img.channels());
        int nch = img.channels();

        // extract padding amounts
        int k_pad = pads.size();
        int pt = 0, pb = 0, pl = 0, pr = 0;
        if (1 == k_pad)
            pt = pb = pl = pr = pads[0];
        else if (2 == k_pad)
        {
            pt = pb = pads[0];
            pl = pr = pads[1];
        }
        else if (4 == k_pad)
        {
            pt = pads[0]; pb = pads[1]; pl = pads[2]; pr = pads[3];
        }
        else
            std::cerr << "Input pad must have 1, 2, or 4 values." << std::endl;

        int flag = 0;
        float pad_val = 0.0;
        if ("replicate" == pad_type)
            flag = 1;
        else if ("symmetric" == pad_type)
            flag = 2;
        else if ("circular" == pad_type)
            flag = 3;
        else
        {
            flag = 0;
            pad_val = 0.1;
        }

        if (0 == ns[0] || 0 == ns[1])
            flag = 0;

        std::vector<int> ms;
        ms.push_back(ns[0] + pt + pb);
        ms.push_back(ns[1] + pl + pr);
        ms.push_back(nch);

        if (ms[0] < 0 || ns[0] <= -pt || ns[0] <= -pb)
            ms[0] = 0;
        if (ms[1] < 0 || ns[1] <= -pl || ns[1] <= -pr)
            ms[1] = 0;

        //cv::Mat padded_img;
        //if (1 == nch)
        //    padded_img = cv::Mat(ms[0], ms[1], CV_32F);
        //if (3 == nch)
        //    padded_img = cv::Mat(ms[0], ms[1], CV_32FC3);
        cv::Mat padded_img(ms[0], ms[1], CV_32FC3);

        cv::Mat img_chan[3];
        cv::split(img, img_chan);

        //cv::Mat img_merge; // stack the channels into a new mat:
        //for (int i_ch = 0; i_ch < 3; ++i_ch)
        //    img_merge.push_back(img_chan[i_ch].t());
        for (int i_ch = 0; i_ch < 3; ++i_ch)
            cv::transpose(img_chan[i_ch], img_chan[i_ch]);
        cv::Mat img_merge = cv::Mat(img.cols * 3, img.rows, CV_32F);
        std::memcpy(img_merge.data + 0 * img.cols * img.rows * img.elemSize1(),
            img_chan[2].data, img.cols * img.rows * img.elemSize1());
        std::memcpy(img_merge.data + 1 * img.cols * img.rows * img.elemSize1(),
            img_chan[1].data, img.cols * img.rows * img.elemSize1());

        std::memcpy(img_merge.data + 2 * img.cols * img.rows * img.elemSize1(),
            img_chan[0].data, img.cols * img.rows * img.elemSize1());
        img_merge = img_merge.reshape(1, 1);

        imPad((float *)img_merge.data, (float *)padded_img.data, ns[0], ns[1], nch,
            pt, pb, pl, pr, flag, pad_val);

        cv::Mat split_pad_img[3];
        for (int i_img = 0; i_img < 3; ++i_img)
        {
            split_pad_img[i_img] = cv::Mat(ms[1], ms[0], CV_32FC1);
        }
        std::memcpy(split_pad_img[2].data, padded_img.data + 0 * ms[0] * ms[1] * padded_img.elemSize1(),
            ms[0] * ms[1] * padded_img.elemSize1());
        std::memcpy(split_pad_img[1].data, padded_img.data + 1 * ms[0] * ms[1] * padded_img.elemSize1(),
            ms[0] * ms[1] * padded_img.elemSize1());
        std::memcpy(split_pad_img[0].data, padded_img.data + 2 * ms[0] * ms[1] * padded_img.elemSize1(),
            ms[0] * ms[1] * padded_img.elemSize1());

        cv::merge(split_pad_img, 3, padded_img);
        cv::transpose(padded_img, padded_img);
        return padded_img;
    }

    __m128 SET(const float &x) { return _mm_set1_ps(x); }
    __m128 SET(float x, float y, float z, float w){ return _mm_set_ps(x, y, z, w); }
    __m128i SET(const int &x) { return _mm_set1_epi32(x); }
    __m128 LD(const float &x) { return _mm_load_ps(&x); }
    __m128 LDu(const float &x) { return _mm_loadu_ps(&x); }
    __m128 STR(float &x, const __m128 y) { _mm_store_ps(&x, y); return y; }
    __m128 STR1(float &x, const __m128 y) { _mm_store_ss(&x, y); return y; }
    __m128 STRu(float &x, const __m128 y) { _mm_storeu_ps(&x, y); return y; }
    __m128 STR(float &x, const float y) { return STR(x, SET(y)); }

    __m128i ADD(const __m128i x, const __m128i y) { return _mm_add_epi32(x, y); }
    __m128 ADD(const __m128 x, const __m128 y) { return _mm_add_ps(x, y); }
    __m128 ADD(const __m128 x, const __m128 y, const __m128 z) {
        return ADD(ADD(x, y), z);
    }
    __m128 ADD(const __m128 a, const __m128 b, const __m128 c, const __m128 &d) {
        return ADD(ADD(ADD(a, b), c), d);
    }
    __m128 SUB(const __m128 x, const __m128 y) { return _mm_sub_ps(x, y); }
    __m128 MUL(const __m128 x, const __m128 y) { return _mm_mul_ps(x, y); }
    __m128 MUL(const __m128 x, const float y) { return MUL(x, SET(y)); }
    __m128 MUL(const float x, const __m128 y) { return MUL(SET(x), y); }
    __m128 INC(__m128 &x, const __m128 y) { return x = ADD(x, y); }
    __m128 INC(float &x, const __m128 y) { __m128 t = ADD(LD(x), y); return STR(x, t); }
    __m128 DEC(__m128 &x, const __m128 y) { return x = SUB(x, y); }
    __m128 DEC(float &x, const __m128 y) { __m128 t = SUB(LD(x), y); return STR(x, t); }
    __m128 MIN1(const __m128 x, const __m128 y) { return _mm_min_ps(x, y); }
    __m128 FMIN(const __m128 x, const __m128 y) { return _mm_min_ps(x, y); }

    __m128 RCP(const __m128 x) { return _mm_rcp_ps(x); }
    __m128 RCPSQRT(const __m128 x) { return _mm_rsqrt_ps(x); }

    // logical operators
    __m128 AND(const __m128 x, const __m128 y) { return _mm_and_ps(x, y); }
    __m128i AND(const __m128i x, const __m128i y) { return _mm_and_si128(x, y); }
    __m128 ANDNOT(const __m128 x, const __m128 y) { return _mm_andnot_ps(x, y); }
    __m128 OR(const __m128 x, const __m128 y) { return _mm_or_ps(x, y); }
    __m128 XOR(const __m128 x, const __m128 y) { return _mm_xor_ps(x, y); }

    // comparison operators
    __m128 CMPGT(const __m128 x, const __m128 y) { return _mm_cmpgt_ps(x, y); }
    __m128 CMPLT(const __m128 x, const __m128 y) { return _mm_cmplt_ps(x, y); }
    __m128i CMPGT(const __m128i x, const __m128i y) { return _mm_cmpgt_epi32(x, y); }
    __m128i CMPLT(const __m128i x, const __m128i y) { return _mm_cmplt_epi32(x, y); }

    // conversion operators
    __m128 CVT(const __m128i x) { return _mm_cvtepi32_ps(x); }
    __m128i CVT(const __m128 x) { return _mm_cvttps_epi32(x); }


// convConstMex.cpp functions
void convBoxY(float *I, float *O, int h, int r, int s) 
{
    float t; int j, p = r + 1, q = 2 * h - (r + 1), h0 = r + 1, h1 = h - r, h2 = h;
    t = 0; for (j = 0; j <= r; j++) t += I[j]; t = 2 * t - I[r]; j = 0;
    if (s == 1) {
        for (; j<h0; j++) O[j] = t -= I[r - j] - I[r + j];
        for (; j<h1; j++) O[j] = t -= I[j - p] - I[r + j];
        for (; j<h2; j++) O[j] = t -= I[j - p] - I[q - j];
    }
    else {
        int k = (s - 1) / 2; h2 = (h / s)*s; if (h0>h2) h0 = h2; if (h1>h2) h1 = h2;
        for (; j<h0; j++) { t -= I[r - j] - I[r + j]; k++; if (k == s) { k = 0; *O++ = t; } }
        for (; j<h1; j++) { t -= I[j - p] - I[r + j]; k++; if (k == s) { k = 0; *O++ = t; } }
        for (; j<h2; j++) { t -= I[j - p] - I[q - j]; k++; if (k == s) { k = 0; *O++ = t; } }
    }
}
void convBox(float *I, float *O, int h, int w, int d, int r, int s) 
{
    float nrm = 1.0f / ((2 * r + 1)*(2 * r + 1)); int i, j, k = (s - 1) / 2, h0, h1, w0;
    if (h % 4 == 0) h0 = h1 = h; else { h0 = h - (h % 4); h1 = h0 + 4; } w0 = (w / s)*s;
    float *T = (float*)alMalloc(h1*sizeof(float), 16);
    while (d-- > 0) {
        // initialize T
        memset(T, 0, h1*sizeof(float));
        for (i = 0; i <= r; i++) for (j = 0; j<h0; j += 4) INC(T[j], LDu(I[j + i*h]));
        for (j = 0; j<h0; j += 4) STR(T[j], MUL(nrm, SUB(MUL(2, LD(T[j])), LDu(I[j + r*h]))));
        for (i = 0; i <= r; i++) for (j = h0; j<h; j++) T[j] += I[j + i*h];
        for (j = h0; j<h; j++) T[j] = nrm*(2 * T[j] - I[j + r*h]);
        // prepare and convolve each column in turn
        k++; if (k == s) { k = 0; convBoxY(T, O, h, r, s); O += h / s; }
        for (i = 1; i<w0; i++) {
            float *Il = I + (i - 1 - r)*h; if (i <= r) Il = I + (r - i)*h;
            float *Ir = I + (i + r)*h; if (i >= w - r) Ir = I + (2 * w - r - i - 1)*h;
            for (j = 0; j<h0; j += 4) DEC(T[j], MUL(nrm, SUB(LDu(Il[j]), LDu(Ir[j]))));
            for (j = h0; j<h; j++) T[j] -= nrm*(Il[j] - Ir[j]);
            k++; if (k == s) { k = 0; convBoxY(T, O, h, r, s); O += h / s; }
        }
        I += w*h;
    }
    alFree(T);
}
void conv11Y(float *I, float *O, int h, int side, int s) 
{
#define C4(m,o) ADD(LDu(I[m*j-1+o]),LDu(I[m*j+o]))
    int j = 0, k = ((~((size_t)O) + 1) & 15) / 4;
    const int d = (side % 4 >= 2) ? 1 : 0, h2 = (h - d) / 2;
    if (s == 2) {
        for (; j<k; j++) O[j] = I[2 * j + d] + I[2 * j + d + 1];
        for (; j<h2 - 4; j += 4) STR(O[j], _mm_shuffle_ps(C4(2, d + 1), C4(2, d + 5), 136));
        for (; j<h2; j++) O[j] = I[2 * j + d] + I[2 * j + d + 1];
        if (d == 1 && h % 2 == 0) O[j] = 2 * I[2 * j + d];
    }
    else {
        if (d == 0) { O[0] = 2 * I[0]; j++; if (k == 0) k = 4; }
        for (; j<k; j++) O[j] = I[j - 1 + d] + I[j + d];
        for (; j<h - 4 - d; j += 4) STR(O[j], C4(1, d));
        for (; j<h - d; j++) O[j] = I[j - 1 + d] + I[j + d];
        if (d == 1) { O[j] = 2 * I[j]; j++; }
    }
#undef C4
}
void conv11(float *I, float *O, int h, int w, int d, int side, int s) 
{
    const float nrm = 0.25f; int i, j;
    float *I0, *I1, *T = (float*)alMalloc(h*sizeof(float), 16);
    for (int d0 = 0; d0<d; d0++) for (i = s / 2; i<w; i += s) {
        I0 = I1 = I + i*h + d0*h*w; if (side % 2) { if (i<w - 1) I1 += h; }
        else { if (i) I0 -= h; }
        for (j = 0; j<h - 4; j += 4) STR(T[j], MUL(nrm, ADD(LDu(I0[j]), LDu(I1[j]))));
        for (; j<h; j++) T[j] = nrm*(I0[j] + I1[j]);
        conv11Y(T, O, h, side, s); O += h / s;
    }
    alFree(T);
}
void convTriY(float *I, float *O, int h, int r, int s) 
{
    r++; float t, u; int j, r0 = r - 1, r1 = r + 1, r2 = 2 * h - r, h0 = r + 1, h1 = h - r + 1, h2 = h;
    u = t = I[0]; for (j = 1; j<r; j++) u += t += I[j]; u = 2 * u - t; t = 0;
    if (s == 1) {
        O[0] = u; j = 1;
        for (; j<h0; j++) O[j] = u += t += I[r - j] + I[r0 + j] - 2 * I[j - 1];
        for (; j<h1; j++) O[j] = u += t += I[j - r1] + I[r0 + j] - 2 * I[j - 1];
        for (; j<h2; j++) O[j] = u += t += I[j - r1] + I[r2 - j] - 2 * I[j - 1];
    }
    else {
        int k = (s - 1) / 2; h2 = (h / s)*s; if (h0>h2) h0 = h2; if (h1>h2) h1 = h2;
        if (++k == s) { k = 0; *O++ = u; } j = 1;
        for (; j<h0; j++) { u += t += I[r - j] + I[r0 + j] - 2 * I[j - 1]; if (++k == s){ k = 0; *O++ = u; } }
        for (; j<h1; j++) { u += t += I[j - r1] + I[r0 + j] - 2 * I[j - 1]; if (++k == s){ k = 0; *O++ = u; } }
        for (; j<h2; j++) { u += t += I[j - r1] + I[r2 - j] - 2 * I[j - 1]; if (++k == s){ k = 0; *O++ = u; } }
    }
}
void convTri(float *I, float *O, int h, int w, int d, int r, int s) 
{
    r++; float nrm = 1.0f / (r*r*r*r); int i, j, k = (s - 1) / 2, h0, h1, w0;
    if (h % 4 == 0) h0 = h1 = h; else { h0 = h - (h % 4); h1 = h0 + 4; } w0 = (w / s)*s;
    float *T = (float*)alMalloc(2 * h1*sizeof(float), 16), *U = T + h1;
    while (d-- > 0) {
        // initialize T and U
        for (j = 0; j<h0; j += 4) STR(U[j], STR(T[j], LDu(I[j])));
        for (i = 1; i<r; i++) for (j = 0; j<h0; j += 4) INC(U[j], INC(T[j], LDu(I[j + i*h])));
        for (j = 0; j<h0; j += 4) STR(U[j], MUL(nrm, (SUB(MUL(2, LD(U[j])), LD(T[j])))));
        for (j = 0; j<h0; j += 4) STR(T[j], 0);
        for (j = h0; j<h; j++) U[j] = T[j] = I[j];
        for (i = 1; i<r; i++) for (j = h0; j<h; j++) U[j] += T[j] += I[j + i*h];
        for (j = h0; j<h; j++) { U[j] = nrm * (2 * U[j] - T[j]); T[j] = 0; }
        // prepare and convolve each column in turn
        k++; if (k == s) { k = 0; convTriY(U, O, h, r - 1, s); O += h / s; }
        for (i = 1; i<w0; i++) {
            float *Il = I + (i - 1 - r)*h; if (i <= r) Il = I + (r - i)*h; float *Im = I + (i - 1)*h;
            float *Ir = I + (i - 1 + r)*h; if (i>w - r) Ir = I + (2 * w - r - i)*h;
            for (j = 0; j<h0; j += 4) {
                INC(T[j], ADD(LDu(Il[j]), LDu(Ir[j]), MUL(-2, LDu(Im[j]))));
                INC(U[j], MUL(nrm, LD(T[j])));
            }
            for (j = h0; j<h; j++) U[j] += nrm*(T[j] += Il[j] + Ir[j] - 2 * Im[j]);
            k++; if (k == s) { k = 0; convTriY(U, O, h, r - 1, s); O += h / s; }
        }
        I += w*h;
    }
    alFree(T);
}
void convTri1Y(float *I, float *O, int h, float p, int s)
{
#define C4(m,o) ADD(ADD(LDu(I[m*j-1+o]),MUL(p,LDu(I[m*j+o]))),LDu(I[m*j+1+o]))
    int j = 0, k = ((~((size_t)O) + 1) & 15) / 4, h2 = (h - 1) / 2;
    if (s == 2) {
        for (; j<k; j++) O[j] = I[2 * j] + p*I[2 * j + 1] + I[2 * j + 2];
        for (; j<h2 - 4; j += 4) STR(O[j], _mm_shuffle_ps(C4(2, 1), C4(2, 5), 136));
        for (; j<h2; j++) O[j] = I[2 * j] + p*I[2 * j + 1] + I[2 * j + 2];
        if (h % 2 == 0) O[j] = I[2 * j] + (1 + p)*I[2 * j + 1];
    }
    else {
        O[j] = (1 + p)*I[j] + I[j + 1]; j++; if (k == 0) k = (h <= 4) ? h - 1 : 4;
        for (; j<k; j++) O[j] = I[j - 1] + p*I[j] + I[j + 1];
        for (; j<h - 4; j += 4) STR(O[j], C4(1, 0));
        for (; j<h - 1; j++) O[j] = I[j - 1] + p*I[j] + I[j + 1];
        O[j] = I[j - 1] + (1 + p)*I[j];
    }
#undef C4
}
void convTri1(float *I, float *O, int h, int w, int d, float p, int s) {
    const float nrm = 1.0f / ((p + 2)*(p + 2)); int i, j, h0 = h - (h % 4);
    float *Il, *Im, *Ir, *T = (float*)alMalloc(h*sizeof(float), 16);
    for (int d0 = 0; d0<d; d0++) for (i = s / 2; i<w; i += s) {
        Il = Im = Ir = I + i*h + d0*h*w; if (i>0) Il -= h; if (i<w - 1) Ir += h;
        for (j = 0; j<h0; j += 4)
            STR(T[j], MUL(nrm, ADD(ADD(LDu(Il[j]), MUL(p, LDu(Im[j]))), LDu(Ir[j]))));
        for (j = h0; j<h; j++) T[j] = nrm*(Il[j] + p*Im[j] + Ir[j]);
        convTri1Y(T, O, h, p, s); O += h / s;
    }
    alFree(T);
}
void convMaxY(float *I, float *O, float *T, int h, int r) {
    int y, y0, y1, yi, m = 2 * r + 1;
#define max1(a,b) a>b ? a : b;
#define maxk(y0,y1) { O[y]=I[y0]; \
    for (yi = y0 + 1; yi <= y1; yi++) { if (I[yi]>O[y]) O[y] = I[yi]; }}
    for (y = 0; y<r; y++) { y1 = y + r; if (y1>h - 1) y1 = h - 1; maxk(0, y1); }
    for (; y <= h - m - r; y += m) {
        T[m - 1] = I[y + r];
        for (yi = 1; yi<m; yi++) T[m - 1 - yi] = max1(T[m - 1 - yi + 1], I[y + r - yi]);
        for (yi = 1; yi<m; yi++) T[m - 1 + yi] = max1(T[m - 1 + yi - 1], I[y + r + yi]);
        for (yi = 0; yi<m; yi++) O[y + yi] = max1(T[yi], T[yi + m - 1]);
    }
    for (; y<h - r; y++) { maxk(y - r, y + r); }
    for (; y<h; y++) { y0 = y - r; if (y0<0) y0 = 0; maxk(y0, h - 1); }
#undef maxk
#undef max1
}
void convMax(float *I, float *O, int h, int w, int d, int r) {
    if (r>w - 1) r = w - 1; if (r>h - 1) r = h - 1; int m = 2 * r + 1;
    float *T = (float*)alMalloc(m * 2 * sizeof(float), 16);
    for (int d0 = 0; d0<d; d0++) for (int x = 0; x<w; x++) {
        float *Oc = O + d0*h*w + h*x, *Ic = I + d0*h*w + h*x;
        convMaxY(Ic, Oc, T, h, r);
    }
    alFree(T);
}
cv::Mat convConstMex(std::string type, cv::Mat img, int r, int s)
{
    int nDims = img.dims; // nDims equals to 2 or 13;
    std::vector<int> ns;
    if (nDims == 2)
    {
        ns.push_back(img.rows);
        ns.push_back(img.cols);
        ns.push_back(img.channels());
    }
    else
    {
        ns.push_back(img.size[1]);
        ns.push_back(img.size[2]);
        ns.push_back(13);
    }
   
    int d = (nDims == 3) ? ns[2] : 1; //dims = 2, d = 1; dims = 3, d = 13;
    int m = (ns[0] < ns[1]) ? ns[0] : ns[1];
    float p = r;

    std::vector<int> ms;
    ms.push_back(ns[0] / s);
    ms.push_back(ns[1] / s);
   // ms.push_back(d);

    int ms1[3] = { 13, ms[1], ms[0] }; // transpose
    int ms2[3] = { 13, ms[0], ms[1] }; // normal

    //checking input
    /*cv::Mat Test1 = cv::Mat(ms[0], ms[1], CV_32F);
    std::memcpy(Test1.data, img.data + 0 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
    cv::Mat Test2 = cv::Mat(ms[0], ms[1], CV_32F);
    std::memcpy(Test2.data, img.data + 1 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
    cv::Mat Test3 = cv::Mat(ms[0], ms[1], CV_32F);
    std::memcpy(Test3.data, img.data + 2 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
    cv::Mat Test4 = cv::Mat(ms[0], ms[1], CV_32F);
    std::memcpy(Test4.data, img.data + 3 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());*/

    // after checking, this img is totally right


    //creating output img
    cv::Mat conved_img;
    cv::Mat _img;
    if (2 == nDims)
    {
        conved_img = cv::Mat(ms[1], ms[0], CV_32F);
        cv::transpose(img, _img);
    }
        
    if (3 == nDims)
    {
        conved_img = cv::Mat(3, ms1, CV_32F); // define tranposed output
        _img = cv::Mat::zeros(3, ms1, CV_32F); // define transposed input

        int a = img.size[1];
        int b = img.size[2];
        int c = _img.size[1];
        int d = _img.size[2];

        // transpose 3d cv::Mat
        for (size_t i = 0; i < 13; i++)
        {
            for (size_t j = 0; j < img.size[2]; j++)
            {
                for (size_t k = 0; k < img.size[1]; k++)
                {
                    _img.at<float>(i, j, k) = img.at<float>(i, k, j);
                }
            }

        }
        //_img is correct
       /* cv::Mat Test1 = cv::Mat(ms[1], ms[0], CV_32F);
        std::memcpy(Test1.data, _img.data + 4 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
        cv::Mat Test2 = cv::Mat(ms[1], ms[0], CV_32F);
        std::memcpy(Test2.data, _img.data + 5 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
        cv::Mat Test3 = cv::Mat(ms[1], ms[0], CV_32F);
        std::memcpy(Test3.data, _img.data + 6 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
        cv::Mat Test4 = cv::Mat(ms[1], ms[0], CV_32F);
        std::memcpy(Test4.data, _img.data + 7 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
*/
    }

    if (type == "convBox")
    {
        //if (r > m /2 ) mexErrMsgTxt("mask larger than image (r too large)");
        convBox((float *)_img.data, (float *)conved_img.data, ns[0], ns[1], d, r, s);
    }
    else if (type ==  "convTri") {
        //if (r >= m / 2) mexErrMsgTxt("mask larger than image (r too large)");
        convTri((float *)_img.data, (float *)conved_img.data, ns[0], ns[1], d, r, s);
    }
    else if (type == "conv11") {
        //if (s>2) mexErrMsgTxt("conv11 can sample by at most s=2");
        conv11((float *)_img.data, (float *)conved_img.data, ns[0], ns[1], d, r, s);
    }
    else if (type == "convTri1") {
        //if (s>2) mexErrMsgTxt("convTri1 can sample by at most s=2");
        convTri1((float *)_img.data, (float *)conved_img.data, ns[0], ns[1], d, p, s);
    }
    else if (type == "convMax") {
        //if (s>1) mexErrMsgTxt("convMax cannot sample");
        convMax((float *)_img.data, (float *)conved_img.data, ns[0], ns[1], d, r);
    }
    else {
        //mexErrMsgTxt("Invalid type.");
    }

    // check out for conved_img, which is correct
 /*   if (nDims == 3)
    {
        cv::Mat Test1 = cv::Mat(ms[1], ms[0], CV_32F);
        std::memcpy(Test1.data, conved_img.data + 4 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
        cv::Mat Test2 = cv::Mat(ms[1], ms[0], CV_32F);
        std::memcpy(Test2.data, conved_img.data + 5 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
        cv::Mat Test3 = cv::Mat(ms[1], ms[0], CV_32F);
        std::memcpy(Test3.data, conved_img.data + 6 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
        cv::Mat Test4 = cv::Mat(ms[1], ms[0], CV_32F);
        std::memcpy(Test4.data, conved_img.data + 7 * ms[1] * ms[0] * img.elemSize1(), ms[1] * ms[0] * img.elemSize1());
    }*/
    
    if (nDims == 2)
    {
        cv::transpose(conved_img, conved_img);
        return conved_img;
    }
    else
    {
        cv::Mat result = cv::Mat(3, ms2, CV_32F);
        for (size_t i = 0; i < 13; i++)
        {
            for (size_t j = 0; j < conved_img.size[2]; j++)
            {
                for (size_t k = 0; k < conved_img.size[1]; k++)
                {
                    result.at<float>(i, j, k) = conved_img.at<float>(i, k, j);
                }
            }

        }
        return result;
    }
}


//rgbConvertMex.cpp functions
float* rgb2luv_setup(float z, float *mr, float *mg, float *mb,
    float &minu, float &minv, float &un, float &vn)
{
    // set constants for conversion
    const float y0 = (float)((6.0 / 29)*(6.0 / 29)*(6.0 / 29));
    const float a = (float)((29.0 / 3)*(29.0 / 3)*(29.0 / 3));
    un = (float) 0.197833; vn = (float) 0.468331;
    mr[0] = (float) 0.430574*z; mr[1] = (float) 0.222015*z; mr[2] = (float) 0.020183*z;
    mg[0] = (float) 0.341550*z; mg[1] = (float) 0.706655*z; mg[2] = (float) 0.129553*z;
    mb[0] = (float) 0.178325*z; mb[1] = (float) 0.071330*z; mb[2] = (float) 0.939180*z;
    float maxi = (float) 1.0 / 270; minu = -88 * maxi; minv = -134 * maxi;
    // build (padded) lookup table for y->l conversion assuming y in [0,1]
    static float lTable[1064]; static bool lInit = false;
    if (lInit) return lTable; float y, l;
    for (int i = 0; i<1025; i++) {
        y = (float)(i / 1024.0);
        l = y>y0 ? 116 * (float)pow((float)y, 1.0 / 3.0) - 16 : y*a;
        lTable[i] = l*maxi;
    }
    for (int i = 1025; i<1064; i++) lTable[i] = lTable[i - 1];
    lInit = true; return lTable;
}
// Convert from rgb to luv
void rgb2luv(float *I, float *J, int n, float nrm) {
    float minu, minv, un, vn, mr[3], mg[3], mb[3];
    float *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
    float *L = J, *U = L + n, *V = U + n; float *R = I, *G = R + n, *B = G + n;
    for (int i = 0; i<n; i++) {
        float r, g, b, x, y, z, l;
        r = (float)*R++; g = (float)*G++; b = (float)*B++;
        x = mr[0] * r + mg[0] * g + mb[0] * b;
        y = mr[1] * r + mg[1] * g + mb[1] * b;
        z = mr[2] * r + mg[2] * g + mb[2] * b;
        l = lTable[(int)(y * 1024)];
        *(L++) = l; z = 1 / (x + 15 * y + 3 * z + (float)1e-35);
        *(U++) = l * (13 * 4 * x*z - 13 * un) - minu;
        *(V++) = l * (13 * 9 * y*z - 13 * vn) - minv;
    }
}
// Convert from rgb to luv using sse
void rgb2luv_sse(float *I, float *J, int n, float nrm) {
    const int k = 256; float R[k], G[k], B[k];
    //if ((size_t(R) & 15 || size_t(G) & 15 || size_t(B) & 15 || size_t(I) & 15 || size_t(J) & 15)
    //    || n % 4>0) {
    //    rgb2luv(I, J, n, nrm); return;
    //}
    int i = 0, i1, n1; float minu, minv, un, vn, mr[3], mg[3], mb[3];
    float *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
    while (i<n) {
        n1 = i + k; if (n1>n) n1 = n; float *J1 = J + i; float *R1, *G1, *B1;
        // convert to floats (and load input into cache)
        if (typeid(float) != typeid(float)) {
            R1 = R; G1 = G; B1 = B; float *Ri = I + i, *Gi = Ri + n, *Bi = Gi + n;
            for (i1 = 0; i1<(n1 - i); i1++) {
                R1[i1] = (float)*Ri++; G1[i1] = (float)*Gi++; B1[i1] = (float)*Bi++;
            }
        }
        else { R1 = ((float*)I) + i; G1 = R1 + n; B1 = G1 + n; }
        // compute RGB -> XYZ
        for (int j = 0; j<3; j++) {
            __m128 _mr, _mg, _mb, *_J = (__m128*) (J1 + j*n);
            __m128 *_R = (__m128*) R1, *_G = (__m128*) G1, *_B = (__m128*) B1;
            _mr = SET(mr[j]); _mg = SET(mg[j]); _mb = SET(mb[j]);
            for (i1 = i; i1<n1; i1 += 4) *(_J++) = ADD(ADD(MUL(*(_R++), _mr),
                MUL(*(_G++), _mg)), MUL(*(_B++), _mb));
        }
        { // compute XZY -> LUV (without doing L lookup/normalization)
            __m128 _c15, _c3, _cEps, _c52, _c117, _c1024, _cun, _cvn;
            _c15 = SET(15.0f); _c3 = SET(3.0f); _cEps = SET(1e-35f);
            _c52 = SET(52.0f); _c117 = SET(117.0f), _c1024 = SET(1024.0f);
            _cun = SET(13 * un); _cvn = SET(13 * vn);
            __m128 *_X, *_Y, *_Z, _x, _y, _z;
            _X = (__m128*) J1; _Y = (__m128*) (J1 + n); _Z = (__m128*) (J1 + 2 * n);
            for (i1 = i; i1<n1; i1 += 4) {
                _x = *_X; _y = *_Y; _z = *_Z;
                _z = RCP(ADD(_x, ADD(_cEps, ADD(MUL(_c15, _y), MUL(_c3, _z)))));
                *(_X++) = MUL(_c1024, _y);
                *(_Y++) = SUB(MUL(MUL(_c52, _x), _z), _cun);
                *(_Z++) = SUB(MUL(MUL(_c117, _y), _z), _cvn);
            }
        }
        { // perform lookup for L and finalize computation of U and V
            for (i1 = i; i1<n1; i1++) J[i1] = lTable[(int)J[i1]];
            __m128 *_L, *_U, *_V, _l, _cminu, _cminv;
            _L = (__m128*) J1; _U = (__m128*) (J1 + n); _V = (__m128*) (J1 + 2 * n);
            _cminu = SET(minu); _cminv = SET(minv);
            for (i1 = i; i1<n1; i1 += 4) {
                _l = *(_L++);
                *_U = SUB(MUL(_l, *_U), _cminu); _U++;
                *_V = SUB(MUL(_l, *_V), _cminv); _V++;
            }
        }
        i = n1;
    }
}
// Convert from rgb to hsv
void rgb2hsv(float *I, float *J, int n, float nrm) {
    float *H = J, *S = H + n, *V = S + n;
    float *R = I, *G = R + n, *B = G + n;
    for (int i = 0; i<n; i++) {
        const float r = (float)*(R++), g = (float)*(G++), b = (float)*(B++);
        float h, s, v, minv, maxv;
        if (r == g && g == b) {
            *(H++) = 0; *(S++) = 0; *(V++) = r*nrm; continue;
        }
        else if (r >= g && r >= b) {
            maxv = r; minv = g<b ? g : b;
            h = (g - b) / (maxv - minv) + 6; if (h >= 6) h -= 6;
        }
        else if (g >= r && g >= b) {
            maxv = g; minv = r<b ? r : b;
            h = (b - r) / (maxv - minv) + 2;
        }
        else {
            maxv = b; minv = r<g ? r : g;
            h = (r - g) / (maxv - minv) + 4;
        }
        h *= (float)(1 / 6.0); s = 1 - minv / maxv; v = maxv*nrm;
        *(H++) = h; *(S++) = s; *(V++) = v;
    }
}
// Convert from rgb to gray
//void rgb2gray(double *I, float *J, int n, float nrm) {
//    float *GR = J; double *R = I, *G = R + n, *B = G + n; int i;
//    float mr = (float).2989360213*nrm, mg = (float).5870430745*nrm, mb = (float).1140209043*nrm;
//    for (i = 0; i<n; i++) *(GR++) = (float)*(R++)*mr + (float)*(G++)*mg + (float)*(B++)*mb;
//}

// Convert from rgb (double) to gray (float)
void rgb2gray(float *I, float *J, int n, float nrm) {
    float *GR = J; float *R = I, *G = R + n, *B = G + n; int i;
    double mr = .2989360213*nrm, mg = .5870430745*nrm, mb = .1140209043*nrm;
    for (i = 0; i<n; i++) *(GR++) = (float)(*(R++)*mr + *(G++)*mg + *(B++)*mb);
}
// Copy and normalize only
void normalize(float *I, float *J, int n, float nrm) {
    for (int i = 0; i<n; i++) *(J++) = (float)*(I++)*nrm;
}
// Convert rgb to various colorspaces
float* rgbConvert(float *I, int n, int d, int flag, float nrm) {
    float *J = (float*)wrMalloc(n*(flag == 0 ? (d == 1 ? 1 : d / 3) : d)*sizeof(float));
    int i, n1 = d*(n<1000 ? n / 10 : 100); float thr = float(1.001);
    if (flag>1 && nrm == 1) for (i = 0; i<n1; i++) if (I[i]>thr)
        wrError("For floats all values in I must be smaller than 1.");
    bool useSse = n % 4 == 0 && typeid(float) == typeid(float);
    if (flag == 2 && useSse)
    for (i = 0; i<d / 3; i++) rgb2luv_sse(I + i*n * 3, (float*)(J + i*n * 3), n, (float)nrm);
    else if ((flag == 0 && d == 1) || flag == 1) normalize(I, J, n*d, nrm);
    else if (flag == 0) for (i = 0; i<d / 3; i++) rgb2gray(I + i*n * 3, J + i*n * 1, n, nrm);
    else if (flag == 2) for (i = 0; i<d / 3; i++) rgb2luv(I + i*n * 3, J + i*n * 3, n, nrm);
    else if (flag == 3) for (i = 0; i<d / 3; i++) rgb2hsv(I + i*n * 3, J + i*n * 3, n, nrm);
    else wrError("Unknown flag.");
    return J;
}
cv::Mat rgbConvertMex(cv::Mat img, int flag, bool useSingle)
{
    std::vector<int> dims;
    dims.push_back(img.rows);
    dims.push_back(img.cols);
    dims.push_back(img.channels());
    int nDims = img.dims;
    int n = dims[0] * dims[1];
    int d = 3;

    cv::Mat img_chan[3];
    cv::split(img, img_chan);

    for (int i_ch = 0; i_ch < 3; ++i_ch)
        cv::transpose(img_chan[i_ch], img_chan[i_ch]);
    cv::Mat img_merge = cv::Mat(img.cols * 3, img.rows, CV_32F);
    std::memcpy(img_merge.data + 0 * img.cols * img.rows * img.elemSize1(),
        img_chan[2].data, img.cols * img.rows * img.elemSize1());
    std::memcpy(img_merge.data + 1 * img.cols * img.rows * img.elemSize1(),
        img_chan[1].data, img.cols * img.rows * img.elemSize1());
    std::memcpy(img_merge.data + 2 * img.cols * img.rows * img.elemSize1(),
        img_chan[0].data, img.cols * img.rows * img.elemSize1());
    img_merge = img_merge.reshape(1, 1);

    // output img
    cv::Mat converted_img;
    if (dims[2] == 3)
    {
        converted_img = cv::Mat(dims[0], dims[1], CV_32FC3);
    }
    else if (dims[2] == 1)
    {
        converted_img = cv::Mat(dims[0], dims[1], CV_32F);
    }
    
    int siz = converted_img.elemSize();
    
    float *J = rgbConvert((float *)img_merge.data, n, d, flag, 1.0f);
   
    std::memcpy(converted_img.data, J, dims[0] * dims[1] * converted_img.elemSize());
    

    cv::Mat split_conv_img[3];
    for (int i_img = 0; i_img < 3; ++i_img)
    {
        split_conv_img[i_img] = cv::Mat(dims[1], dims[0], CV_32FC1);
    }
    std::memcpy(split_conv_img[2].data, converted_img.data + 0 * dims[0] * dims[1] * converted_img.elemSize1(),
        dims[0] * dims[1] * converted_img.elemSize1());
    std::memcpy(split_conv_img[1].data, converted_img.data + 1 * dims[0] * dims[1] * converted_img.elemSize1(),
        dims[0] * dims[1] * converted_img.elemSize1());
    std::memcpy(split_conv_img[0].data, converted_img.data + 2 * dims[0] * dims[1] * converted_img.elemSize1(),
        dims[0] * dims[1] * converted_img.elemSize1());

    cv::merge(split_conv_img, 3, converted_img);
    cv::transpose(converted_img, converted_img);
    return converted_img;   
}


// ImResampleMex.cpp functions
void resampleCoef(int ha, int hb, int &n, int *&yas,
    int *&ybs, float *&wts, int bd[2], int pad)
{
    const float s = float(hb) / float(ha), sInv = 1 / s; float wt, wt0 = float(1e-3)*s;
    bool ds = ha>hb; int nMax; bd[0] = bd[1] = 0;
    if (ds) { n = 0; nMax = ha + (pad>2 ? pad : 2)*hb; }
    else { n = nMax = hb; }
    // initialize memory
    wts = (float*)alMalloc(nMax*sizeof(float), 16);
    yas = (int*)alMalloc(nMax*sizeof(int), 16);
    ybs = (int*)alMalloc(nMax*sizeof(int), 16);
    if (ds) for (int yb = 0; yb<hb; yb++) {
        // create coefficients for downsampling
        float ya0f = yb*sInv, ya1f = ya0f + sInv, W = 0;
        int ya0 = int(ceil(ya0f)), ya1 = int(ya1f), n1 = 0;
        for (int ya = ya0 - 1; ya<ya1 + 1; ya++) {
            wt = s; if (ya == ya0 - 1) wt = (ya0 - ya0f)*s; else if (ya == ya1) wt = (ya1f - ya1)*s;
            if (wt>wt0 && ya >= 0) { ybs[n] = yb; yas[n] = ya; wts[n] = wt; n++; n1++; W += wt; }
        }
        if (W>1) for (int i = 0; i<n1; i++) wts[n - n1 + i] /= W;
        if (n1>bd[0]) bd[0] = n1;
        while (n1<pad) { ybs[n] = yb; yas[n] = yas[n - 1]; wts[n] = 0; n++; n1++; }
    }
    else for (int yb = 0; yb<hb; yb++) {
        // create coefficients for upsampling
        float yaf = (float(.5) + yb)*sInv - float(.5); int ya = (int)floor(yaf);
        wt = 1; if (ya >= 0 && ya<ha - 1) wt = 1 - (yaf - ya);
        if (ya<0) { ya = 0; bd[0]++; } if (ya >= ha - 1) { ya = ha - 1; bd[1]++; }
        ybs[yb] = yb; yas[yb] = ya; wts[yb] = wt;
    }
}
void resample(float *A, float *B, int ha, int hb, int wa, int wb, int d, float(r)) {
    int hn, wn, x, x1, y, z, xa, xb, ya; float *A0, *A1, *A2, *A3, *B0, wt, wt1;
    float *C = (float*)alMalloc((ha + 4)*sizeof(float), 16); for (y = ha; y<ha + 4; y++) C[y] = 0;
    bool sse = (typeid(float) == typeid(float)) && !(size_t(A) & 15) && !(size_t(B) & 15);
    // get coefficients for resampling along w and h
    int *xas, *xbs, *yas, *ybs; float *xwts, *ywts; int xbd[2], ybd[2];
    resampleCoef(wa, wb, wn, xas, xbs, xwts, xbd, 0);
    resampleCoef(ha, hb, hn, yas, ybs, ywts, ybd, 4);
    if (wa == 2 * wb) r /= 2; if (wa == 3 * wb) r /= 3; if (wa == 4 * wb) r /= 4;
    r /= float(1 + 1e-6); for (y = 0; y<hn; y++) ywts[y] *= r;
    // resample each channel in turn
    for (z = 0; z<d; z++) for (x = 0; x<wb; x++) {
        if (x == 0) x1 = 0; xa = xas[x1]; xb = xbs[x1]; wt = xwts[x1]; wt1 = 1 - wt; y = 0;
        A0 = A + z*ha*wa + xa*ha; A1 = A0 + ha, A2 = A1 + ha, A3 = A2 + ha; B0 = B + z*hb*wb + xb*hb;
        // variables for SSE (simple casts to float)
        float *Af0, *Af1, *Af2, *Af3, *Bf0, *Cf, *ywtsf, wtf, wt1f;
        Af0 = (float*)A0; Af1 = (float*)A1; Af2 = (float*)A2; Af3 = (float*)A3;
        Bf0 = (float*)B0; Cf = (float*)C;
        ywtsf = (float*)ywts; wtf = (float)wt; wt1f = (float)wt1;
        // resample along x direction (A -> C)
#define FORs(X) if(sse) for(; y<ha-4; y+=4) STR(Cf[y],X);
#define FORr(X) for(; y<ha; y++) C[y] = X;
        if (wa == 2 * wb) {
            FORs(ADD(LDu(Af0[y]), LDu(Af1[y])));
            FORr(A0[y] + A1[y]); x1 += 2;
        }
        else if (wa == 3 * wb) {
            FORs(ADD(LDu(Af0[y]), LDu(Af1[y]), LDu(Af2[y])));
            FORr(A0[y] + A1[y] + A2[y]); x1 += 3;
        }
        else if (wa == 4 * wb) {
            FORs(ADD(LDu(Af0[y]), LDu(Af1[y]), LDu(Af2[y]), LDu(Af3[y])));
            FORr(A0[y] + A1[y] + A2[y] + A3[y]); x1 += 4;
        }
        else if (wa>wb) {
            int m = 1; while (x1 + m<wn && xb == xbs[x1 + m]) m++; float wtsf[4];
            for (int x0 = 0; x0<(m<4 ? m : 4); x0++) wtsf[x0] = float(xwts[x1 + x0]);
#define U(x) MUL( LDu(*(Af ## x + y)), SET(wtsf[x]) )
#define V(x) *(A ## x + y) * xwts[x1+x]
            if (m == 1) { FORs(U(0));                     FORr(V(0)); }
            if (m == 2) { FORs(ADD(U(0), U(1)));           FORr(V(0) + V(1)); }
            if (m == 3) { FORs(ADD(U(0), U(1), U(2)));      FORr(V(0) + V(1) + V(2)); }
            if (m >= 4) { FORs(ADD(U(0), U(1), U(2), U(3))); FORr(V(0) + V(1) + V(2) + V(3)); }
#undef U
#undef V
            for (int x0 = 4; x0<m; x0++) {
                A1 = A0 + x0*ha; wt1 = xwts[x1 + x0]; Af1 = (float*)A1; wt1f = float(wt1); y = 0;
                FORs(ADD(LD(Cf[y]), MUL(LDu(Af1[y]), SET(wt1f)))); FORr(C[y] + A1[y] * wt1);
            }
            x1 += m;
        }
        else {
            bool xBd = x<xbd[0] || x >= wb - xbd[1]; x1++;
            if (xBd) memcpy(C, A0, ha*sizeof(float));
            if (!xBd) FORs(ADD(MUL(LDu(Af0[y]), SET(wtf)), MUL(LDu(Af1[y]), SET(wt1f))));
            if (!xBd) FORr(A0[y] * wt + A1[y] * wt1);
        }
#undef FORs
#undef FORr
        // resample along y direction (B -> C)
        if (ha == hb * 2) {
            float r2 = r / 2; int k = ((~((size_t)B0) + 1) & 15) / 4; y = 0;
            for (; y<k; y++)  B0[y] = (C[2 * y] + C[2 * y + 1])*r2;
            if (sse) for (; y<hb - 4; y += 4) STR(Bf0[y], MUL((float)r2, _mm_shuffle_ps(ADD(
                LDu(Cf[2 * y]), LDu(Cf[2 * y + 1])), ADD(LDu(Cf[2 * y + 4]), LDu(Cf[2 * y + 5])), 136)));
            for (; y<hb; y++) B0[y] = (C[2 * y] + C[2 * y + 1])*r2;
        }
        else if (ha == hb * 3) {
            for (y = 0; y<hb; y++) B0[y] = (C[3 * y] + C[3 * y + 1] + C[3 * y + 2])*(r / 3);
        }
        else if (ha == hb * 4) {
            for (y = 0; y<hb; y++) B0[y] = (C[4 * y] + C[4 * y + 1] + C[4 * y + 2] + C[4 * y + 3])*(r / 4);
        }
        else if (ha>hb) {
            y = 0;
            //if( sse && ybd[0]<=4 ) for(; y<hb; y++) // Requires SSE4
            //  STR1(Bf0[y],_mm_dp_ps(LDu(Cf[yas[y*4]]),LDu(ywtsf[y*4]),0xF1));
#define U(o) C[ya+o]*ywts[y*4+o]
            if (ybd[0] == 2) for (; y<hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1); }
            if (ybd[0] == 3) for (; y<hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1) + U(2); }
            if (ybd[0] == 4) for (; y<hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1) + U(2) + U(3); }
            if (ybd[0]>4)  for (; y<hn; y++) { B0[ybs[y]] += C[yas[y]] * ywts[y]; }
#undef U
        }
        else {
            for (y = 0; y<ybd[0]; y++) B0[y] = C[yas[y]] * ywts[y];
            for (; y<hb - ybd[1]; y++) B0[y] = C[yas[y]] * ywts[y] + C[yas[y] + 1] * (r - ywts[y]);
            for (; y<hb; y++)        B0[y] = C[yas[y]] * ywts[y];
        }
    }
    alFree(xas); alFree(xbs); alFree(xwts); alFree(C);
    alFree(yas); alFree(ybs); alFree(ywts);
}
//void resample(double *A, double *B, int ha, int hb, int wa, int wb, int d, double r) {
//    int hn, wn, x, x1, y, z, xa, xb, ya; double *A0, *A1, *A2, *A3, *B0, wt, wt1;
//    double *C = (double*)alMalloc((ha + 4)*sizeof(double), 16); for (y = ha; y<ha + 4; y++) C[y] = 0;
//    bool sse = (typeid(double) == typeid(float)) && !(size_t(A) & 15) && !(size_t(B) & 15);
//    // get coefficients for resampling along w and h
//    int *xas, *xbs, *yas, *ybs; double *xwts, *ywts; int xbd[2], ybd[2];
//    resampleCoef<double>(wa, wb, wn, xas, xbs, xwts, xbd, 0);
//    resampleCoef<double>(ha, hb, hn, yas, ybs, ywts, ybd, 4);
//    if (wa == 2 * wb) r /= 2; if (wa == 3 * wb) r /= 3; if (wa == 4 * wb) r /= 4;
//    r /= double(1 + 1e-6); for (y = 0; y<hn; y++) ywts[y] *= r;
//    // resample each channel in turn
//    for (z = 0; z<d; z++) for (x = 0; x<wb; x++) {
//        if (x == 0) x1 = 0; xa = xas[x1]; xb = xbs[x1]; wt = xwts[x1]; wt1 = 1 - wt; y = 0;
//        A0 = A + z*ha*wa + xa*ha; A1 = A0 + ha, A2 = A1 + ha, A3 = A2 + ha; B0 = B + z*hb*wb + xb*hb;
//        // variables for SSE (simple casts to float)
//        float *Af0, *Af1, *Af2, *Af3, *Bf0, *Cf, *ywtsf, wtf, wt1f;
//        Af0 = (float*)A0; Af1 = (float*)A1; Af2 = (float*)A2; Af3 = (float*)A3;
//        Bf0 = (float*)B0; Cf = (float*)C;
//        ywtsf = (float*)ywts; wtf = (float)wt; wt1f = (float)wt1;
//        // resample along x direction (A -> C)
//#define FORs(X) if(sse) for(; y<ha-4; y+=4) STR(Cf[y],X);
//#define FORr(X) for(; y<ha; y++) C[y] = X;
//        if (wa == 2 * wb) {
//            FORs(ADD(LDu(Af0[y]), LDu(Af1[y])));
//            FORr(A0[y] + A1[y]); x1 += 2;
//        }
//        else if (wa == 3 * wb) {
//            FORs(ADD(LDu(Af0[y]), LDu(Af1[y]), LDu(Af2[y])));
//            FORr(A0[y] + A1[y] + A2[y]); x1 += 3;
//        }
//        else if (wa == 4 * wb) {
//            FORs(ADD(LDu(Af0[y]), LDu(Af1[y]), LDu(Af2[y]), LDu(Af3[y])));
//            FORr(A0[y] + A1[y] + A2[y] + A3[y]); x1 += 4;
//        }
//        else if (wa>wb) {
//            int m = 1; while (x1 + m<wn && xb == xbs[x1 + m]) m++; float wtsf[4];
//            for (int x0 = 0; x0<(m<4 ? m : 4); x0++) wtsf[x0] = float(xwts[x1 + x0]);
//#define U(x) MUL( LDu(*(Af ## x + y)), SET(wtsf[x]) )
//#define V(x) *(A ## x + y) * xwts[x1+x]
//            if (m == 1) { FORs(U(0));                     FORr(V(0)); }
//            if (m == 2) { FORs(ADD(U(0), U(1)));           FORr(V(0) + V(1)); }
//            if (m == 3) { FORs(ADD(U(0), U(1), U(2)));      FORr(V(0) + V(1) + V(2)); }
//            if (m >= 4) { FORs(ADD(U(0), U(1), U(2), U(3))); FORr(V(0) + V(1) + V(2) + V(3)); }
//#undef U
//#undef V
//            for (int x0 = 4; x0<m; x0++) {
//                A1 = A0 + x0*ha; wt1 = xwts[x1 + x0]; Af1 = (float*)A1; wt1f = float(wt1); y = 0;
//                FORs(ADD(LD(Cf[y]), MUL(LDu(Af1[y]), SET(wt1f)))); FORr(C[y] + A1[y] * wt1);
//            }
//            x1 += m;
//        }
//        else {
//            bool xBd = x<xbd[0] || x >= wb - xbd[1]; x1++;
//            if (xBd) memcpy(C, A0, ha*sizeof(double));
//            if (!xBd) FORs(ADD(MUL(LDu(Af0[y]), SET(wtf)), MUL(LDu(Af1[y]), SET(wt1f))));
//            if (!xBd) FORr(A0[y] * wt + A1[y] * wt1);
//        }
//#undef FORs
//#undef FORr
//        // resample along y direction (B -> C)
//        if (ha == hb * 2) {
//            double r2 = r / 2; int k = ((~((size_t)B0) + 1) & 15) / 4; y = 0;
//            for (; y<k; y++)  B0[y] = (C[2 * y] + C[2 * y + 1])*r2;
//            if (sse) for (; y<hb - 4; y += 4) STR(Bf0[y], MUL((float)r2, _mm_shuffle_ps(ADD(
//                LDu(Cf[2 * y]), LDu(Cf[2 * y + 1])), ADD(LDu(Cf[2 * y + 4]), LDu(Cf[2 * y + 5])), 136)));
//            for (; y<hb; y++) B0[y] = (C[2 * y] + C[2 * y + 1])*r2;
//        }
//        else if (ha == hb * 3) {
//            for (y = 0; y<hb; y++) B0[y] = (C[3 * y] + C[3 * y + 1] + C[3 * y + 2])*(r / 3);
//        }
//        else if (ha == hb * 4) {
//            for (y = 0; y<hb; y++) B0[y] = (C[4 * y] + C[4 * y + 1] + C[4 * y + 2] + C[4 * y + 3])*(r / 4);
//        }
//        else if (ha>hb) {
//            y = 0;
//            //if( sse && ybd[0]<=4 ) for(; y<hb; y++) // Requires SSE4
//            //  STR1(Bf0[y],_mm_dp_ps(LDu(Cf[yas[y*4]]),LDu(ywtsf[y*4]),0xF1));
//#define U(o) C[ya+o]*ywts[y*4+o]
//            if (ybd[0] == 2) for (; y<hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1); }
//            if (ybd[0] == 3) for (; y<hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1) + U(2); }
//            if (ybd[0] == 4) for (; y<hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1) + U(2) + U(3); }
//            if (ybd[0]>4)  for (; y<hn; y++) { B0[ybs[y]] += C[yas[y]] * ywts[y]; }
//#undef U
//        }
//        else {
//            for (y = 0; y<ybd[0]; y++) B0[y] = C[yas[y]] * ywts[y];
//            for (; y<hb - ybd[1]; y++) B0[y] = C[yas[y]] * ywts[y] + C[yas[y] + 1] * (r - ywts[y]);
//            for (; y<hb; y++)        B0[y] = C[yas[y]] * ywts[y];
//        }
//    }
//    alFree(xas); alFree(xbs); alFree(xwts); alFree(C);
//    alFree(yas); alFree(ybs); alFree(ywts);
//}
cv::Mat ImResampleMex(cv::Mat img, int height, int weight, int norm)
{
    int nDims = img.dims;
    std::vector<int> ns;
    ns.push_back(img.rows);
    ns.push_back(img.cols);
    ns.push_back(img.channels());
    int nCh = (nDims == 2) ? 1 : ns[2];
    std::vector<int> ms;
    ms.push_back(height);
    ms.push_back(weight);
    ms.push_back(nCh);
    double nrm = norm;

    if (img.channels() == 1)
    {
        cv::transpose(img, img);
        cv::Mat sampled_img(ms[1], ms[0], CV_32FC1);
        resample((float *)img.data, (float *)sampled_img.data, ns[0], ms[0], ns[1], ms[1], nCh, float(nrm));
        cv::transpose(sampled_img, sampled_img);
        return sampled_img;
    }
    else if (img.channels() == 3)
    {
        nCh = 3; // manually set nCh = 3 
        cv::Mat img_chan[3];
        cv::split(img, img_chan);

        // transform input img
        for (int i_ch = 0; i_ch < 3; ++i_ch)
            cv::transpose(img_chan[i_ch], img_chan[i_ch]);
        cv::Mat img_merge = cv::Mat(img.cols * 3, img.rows, CV_32F);
        std::memcpy(img_merge.data + 0 * img.cols * img.rows * img.elemSize1(),
            img_chan[2].data, img.cols * img.rows * img.elemSize1());
        std::memcpy(img_merge.data + 1 * img.cols * img.rows * img.elemSize1(),
            img_chan[1].data, img.cols * img.rows * img.elemSize1());
        std::memcpy(img_merge.data + 2 * img.cols * img.rows * img.elemSize1(),
            img_chan[0].data, img.cols * img.rows * img.elemSize1());
        img_merge = img_merge.reshape(1, 1);

        cv::Mat sampled_img(ms[1], ms[0], CV_32FC3);

        resample((float*)img_merge.data, (float*)sampled_img.data, ns[0], ms[0], ns[1], ms[1], nCh, float(nrm));

        cv::Mat split_pad_img[3];
        for (int i_img = 0; i_img < 3; ++i_img)
        {
            split_pad_img[i_img] = cv::Mat(ms[1], ms[0], CV_32FC1);
        }
        std::memcpy(split_pad_img[2].data, sampled_img.data + 0 * ms[0] * ms[1] * sampled_img.elemSize1(),
            ms[0] * ms[1] * sampled_img.elemSize1());
        std::memcpy(split_pad_img[1].data, sampled_img.data + 1 * ms[0] * ms[1] * sampled_img.elemSize1(),
            ms[0] * ms[1] * sampled_img.elemSize1());
        std::memcpy(split_pad_img[0].data, sampled_img.data + 2 * ms[0] * ms[1] * sampled_img.elemSize1(),
            ms[0] * ms[1] * sampled_img.elemSize1());

        cv::merge(split_pad_img, 3, sampled_img);
        cv::transpose(sampled_img, sampled_img);
        return sampled_img;
    }
}

// GradientMex functions
void grad1(float *I, float *Gx, float *Gy, int h, int w, int x) {
    int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
    // compute column of Gx
    Ip = I - h; In = I + h; r = .5f;
    if (x == 0) { r = 1; Ip += h; }
    else if (x == w - 1) { r = 1; In -= h; }
    if (h<4 || h % 4>0 || (size_t(I) & 15) || (size_t(Gx) & 15)) {
        for (y = 0; y<h; y++) *Gx++ = (*In++ - *Ip++)*r;
    }
    else {
        _G = (__m128*) Gx; _Ip = (__m128*) Ip; _In = (__m128*) In; _r = SET(r);
        for (y = 0; y<h; y += 4) *_G++ = MUL(SUB(*_In++, *_Ip++), _r);
    }
    // compute column of Gy
#define GRADY(r) *Gy++=(*In++-*Ip++)*r;
    Ip = I; In = Ip + 1;
    // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
    y1 = ((~((size_t)Gy) + 1) & 15) / 4; if (y1 == 0) y1 = 4; if (y1>h - 1) y1 = h - 1;
    GRADY(1); Ip--; for (y = 1; y<y1; y++) GRADY(.5f);
    _r = SET(.5f); _G = (__m128*) Gy;
    for (; y + 4<h - 1; y += 4, Ip += 4, In += 4, Gy += 4)
        *_G++ = MUL(SUB(LDu(*In), LDu(*Ip)), _r);
    for (; y<h - 1; y++) GRADY(.5f); In--; GRADY(1);
#undef GRADY
}

// compute x and y gradients at each location (uses sse)
void grad2(float *I, float *Gx, float *Gy, int h, int w, int d) {
    int o, x, c, a = w*h; for (c = 0; c<d; c++) for (x = 0; x<w; x++) {
        o = c*a + x*h; grad1(I + o, Gx + o, Gy + o, h, w, x);
    }
}

// build lookup table a[] s.t. a[x*n]~=acos(x) for x in [-1,1]
float* acosTable() {
    const int n = 10000, b = 10; int i;
    static float a[n * 2 + b * 2]; static bool init = false;
    float *a1 = a + n + b; if (init) return a1;
    for (i = -n - b; i<-n; i++)   a1[i] = PI;
    for (i = -n; i<n; i++)      a1[i] = float(acos(i / float(n)));
    for (i = n; i<n + b; i++)     a1[i] = 0;
    for (i = -n - b; i<n / 10; i++) if (a1[i] > PI - 1e-6f) a1[i] = PI - 1e-6f;
    init = true; return a1;
}

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full) {
    int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
    float *acost = acosTable(), acMult = 10000.0f;
    // allocate memory for storing one column of output (padded so h4%4==0)
    h4 = (h % 4 == 0) ? h : h - (h % 4) + 4; s = d*h4*sizeof(float);
    M2 = (float*)alMalloc(s, 16); _M2 = (__m128*) M2;
    Gx = (float*)alMalloc(s, 16); _Gx = (__m128*) Gx;
    Gy = (float*)alMalloc(s, 16); _Gy = (__m128*) Gy;
    // compute gradient magnitude and orientation for each column
    for (x = 0; x<w; x++) {
        // compute gradients (Gx, Gy) with maximum squared magnitude (M2)
        for (c = 0; c<d; c++) {
            grad1(I + x*h + c*w*h, Gx + c*h4, Gy + c*h4, h, w, x);
            for (y = 0; y<h4 / 4; y++) {
                y1 = h4 / 4 * c + y;
                _M2[y1] = ADD(MUL(_Gx[y1], _Gx[y1]), MUL(_Gy[y1], _Gy[y1]));
                if (c == 0) continue; _m = CMPGT(_M2[y1], _M2[y]);
                _M2[y] = OR(AND(_m, _M2[y1]), ANDNOT(_m, _M2[y]));
                _Gx[y] = OR(AND(_m, _Gx[y1]), ANDNOT(_m, _Gx[y]));
                _Gy[y] = OR(AND(_m, _Gy[y1]), ANDNOT(_m, _Gy[y]));
            }
        }
        // compute gradient mangitude (M) and normalize Gx
        for (y = 0; y<h4 / 4; y++) {
            _m = MIN1(RCPSQRT(_M2[y]), SET(1e10f));
            _M2[y] = RCP(_m);
            if (O) _Gx[y] = MUL(MUL(_Gx[y], _m), SET(acMult));
            if (O) _Gx[y] = XOR(_Gx[y], AND(_Gy[y], SET(-0.f)));
        };
        memcpy(M + x*h, M2, h*sizeof(float));
        // compute and store gradient orientation (O) via table lookup
        if (O != 0) for (y = 0; y<h; y++) O[x*h + y] = acost[(int)Gx[y]];
        if (O != 0 && full) {
            y1 = ((~size_t(O + x*h) + 1) & 15) / 4; y = 0;
            for (; y<y1; y++) O[y + x*h] += (Gy[y]<0)*PI;
            for (; y<h - 4; y += 4) STRu(O[y + x*h],
                ADD(LDu(O[y + x*h]), AND(CMPLT(LDu(Gy[y]), SET(0.f)), SET(PI))));
            for (; y<h; y++) O[y + x*h] += (Gy[y]<0)*PI;
        }
    }
    alFree(Gx); alFree(Gy); alFree(M2);
}

// normalize gradient magnitude at each location (uses sse)
void gradMagNorm(float *M, float *S, int h, int w, float norm) {
    __m128 *_M, *_S, _norm; int i = 0, n = h*w, n4 = n / 4;
    _S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
    bool sse = !(size_t(M) & 15) && !(size_t(S) & 15);
    if (sse) for (; i<n4; i++) { *_M = MUL(*_M, RCP(ADD(*_S++, _norm))); _M++; }
    if (sse) i *= 4; for (; i<n; i++) M[i] /= (S[i] + norm);
}

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1,
    int nb, int n, float norm, int nOrients, bool full, bool interpolate)
{
    // assumes all *OUTPUT* matrices are 4-byte aligned
    int i, o0, o1; float o, od, m;
    __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;
    // define useful constants
    const float oMult = (float)nOrients / (full ? 2 * PI : PI); const int oMax = nOrients*nb;
    const __m128 _norm = SET(norm), _oMult = SET(oMult), _nbf = SET((float)nb);
    const __m128i _oMax = SET(oMax), _nb = SET(nb);
    // perform the majority of the work with sse
    _O0 = (__m128i*) O0; _O1 = (__m128i*) O1; _M0 = (__m128*) M0; _M1 = (__m128*) M1;
    if (interpolate) for (i = 0; i <= n - 4; i += 4) {
        _o = MUL(LDu(O[i]), _oMult); _o0 = CVT(_o); _od = SUB(_o, CVT(_o0));
        _o0 = CVT(MUL(CVT(_o0), _nbf)); _o0 = AND(CMPGT(_oMax, _o0), _o0); *_O0++ = _o0;
        _o1 = ADD(_o0, _nb); _o1 = AND(CMPGT(_oMax, _o1), _o1); *_O1++ = _o1;
        _m = MUL(LDu(M[i]), _norm); *_M1 = MUL(_od, _m); *_M0++ = SUB(_m, *_M1); _M1++;
    }
    else for (i = 0; i <= n - 4; i += 4) {
        _o = MUL(LDu(O[i]), _oMult); _o0 = CVT(ADD(_o, SET(.5f)));
        _o0 = CVT(MUL(CVT(_o0), _nbf)); _o0 = AND(CMPGT(_oMax, _o0), _o0); *_O0++ = _o0;
        *_M0++ = MUL(LDu(M[i]), _norm); *_M1++ = SET(0.f); *_O1++ = SET(0);
    }
    // compute trailing locations without sse
    if (interpolate) for (; i<n; i++) {
        o = O[i] * oMult; o0 = (int)o; od = o - o0;
        o0 *= nb; if (o0 >= oMax) o0 = 0; O0[i] = o0;
        o1 = o0 + nb; if (o1 == oMax) o1 = 0; O1[i] = o1;
        m = M[i] * norm; M1[i] = od*m; M0[i] = m - M1[i];
    }
    else for (; i<n; i++) {
        o = O[i] * oMult; o0 = (int)(o + .5f);
        o0 *= nb; if (o0 >= oMax) o0 = 0; O0[i] = o0;
        M0[i] = M[i] * norm; M1[i] = 0; O1[i] = 0;
    }
}

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist(float *M, float *O, float *H, int h, int w,
    int bin, int nOrients, int softBin, bool full)
{
    const int hb = h / bin, wb = w / bin, h0 = hb*bin, w0 = wb*bin, nb = wb*hb;
    const float s = (float)bin, sInv = 1 / s, sInv2 = 1 / s / s;
    float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1; float xb, init;
    O0 = (int*)alMalloc(h*sizeof(int), 16); M0 = (float*)alMalloc(h*sizeof(float), 16);
    O1 = (int*)alMalloc(h*sizeof(int), 16); M1 = (float*)alMalloc(h*sizeof(float), 16);
    // main loop
    for (x = 0; x<w0; x++) {
        // compute target orientation bins for entire column - very fast
        gradQuantize(O + x*h, M + x*h, O0, O1, M0, M1, nb, h0, sInv2, nOrients, full, softBin >= 0);

        if (softBin<0 && softBin % 2 == 0) {
            // no interpolation w.r.t. either orienation or spatial bin
            H1 = H + (x / bin)*hb;
#define GH H1[O0[y]]+=M0[y]; y++;
            if (bin == 1)      for (y = 0; y<h0;) { GH; H1++; }
            else if (bin == 2) for (y = 0; y<h0;) { GH; GH; H1++; }
            else if (bin == 3) for (y = 0; y<h0;) { GH; GH; GH; H1++; }
            else if (bin == 4) for (y = 0; y<h0;) { GH; GH; GH; GH; H1++; }
            else for (y = 0; y<h0;) { for (int y1 = 0; y1<bin; y1++) { GH; } H1++; }
#undef GH

        }
        else if (softBin % 2 == 0 || bin == 1) {
            // interpolate w.r.t. orientation only, not spatial bin
            H1 = H + (x / bin)*hb;
#define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
            if (bin == 1)      for (y = 0; y<h0;) { GH; H1++; }
            else if (bin == 2) for (y = 0; y<h0;) { GH; GH; H1++; }
            else if (bin == 3) for (y = 0; y<h0;) { GH; GH; GH; H1++; }
            else if (bin == 4) for (y = 0; y<h0;) { GH; GH; GH; GH; H1++; }
            else for (y = 0; y<h0;) { for (int y1 = 0; y1<bin; y1++) { GH; } H1++; }
#undef GH

        }
        else {
            // interpolate using trilinear interpolation
            float ms[4], xyd, yb, xd, yd; __m128 _m, _m0, _m1;
            bool hasLf, hasRt; int xb0, yb0;
            if (x == 0) { init = (0 + .5f)*sInv - 0.5f; xb = init; }
            hasLf = xb >= 0; xb0 = hasLf ? (int)xb : -1; hasRt = xb0 < wb - 1;
            xd = xb - xb0; xb += sInv; yb = init; y = 0;
            // macros for code conciseness
#define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
    ms[0] = 1 - xd - yd + xyd; ms[1] = yd - xyd; ms[2] = xd - xyd; ms[3] = xyd;
#define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));
            // leading rows, no top bin
            for (; y<bin / 2; y++) {
                yb0 = -1; GHinit;
                if (hasLf) { H0[O0[y] + 1] += ms[1] * M0[y]; H0[O1[y] + 1] += ms[1] * M1[y]; }
                if (hasRt) { H0[O0[y] + hb + 1] += ms[3] * M0[y]; H0[O1[y] + hb + 1] += ms[3] * M1[y]; }
            }
            // main rows, has top and bottom bins, use SSE for minor speedup
            if (softBin<0) for (;; y++) {
                yb0 = (int)yb; if (yb0 >= hb - 1) break; GHinit; _m0 = SET(M0[y]);
                if (hasLf) { _m = SET(0, 0, ms[1], ms[0]); GH(H0 + O0[y], _m, _m0); }
                if (hasRt) { _m = SET(0, 0, ms[3], ms[2]); GH(H0 + O0[y] + hb, _m, _m0); }
            }
            else for (;; y++) {
                yb0 = (int)yb; if (yb0 >= hb - 1) break; GHinit;
                _m0 = SET(M0[y]); _m1 = SET(M1[y]);
                if (hasLf) {
                    _m = SET(0, 0, ms[1], ms[0]);
                    GH(H0 + O0[y], _m, _m0); GH(H0 + O1[y], _m, _m1);
                }
                if (hasRt) {
                    _m = SET(0, 0, ms[3], ms[2]);
                    GH(H0 + O0[y] + hb, _m, _m0); GH(H0 + O1[y] + hb, _m, _m1);
                }
            }
            // final rows, no bottom bin
            for (; y<h0; y++) {
                yb0 = (int)yb; GHinit;
                if (hasLf) { H0[O0[y]] += ms[0] * M0[y]; H0[O1[y]] += ms[0] * M1[y]; }
                if (hasRt) { H0[O0[y] + hb] += ms[2] * M0[y]; H0[O1[y] + hb] += ms[2] * M1[y]; }
            }
#undef GHinit
#undef GH
        }
    }
    alFree(O0); alFree(O1); alFree(M0); alFree(M1);
    // normalize boundary bins which only get 7/8 of weight of interior bins
    if (softBin % 2 != 0) for (int o = 0; o<nOrients; o++) {
        x = 0; for (y = 0; y<hb; y++) H[o*nb + x*hb + y] *= 8.f / 7.f;
        y = 0; for (x = 0; x<wb; x++) H[o*nb + x*hb + y] *= 8.f / 7.f;
        x = wb - 1; for (y = 0; y<hb; y++) H[o*nb + x*hb + y] *= 8.f / 7.f;
        y = hb - 1; for (x = 0; x<wb; x++) H[o*nb + x*hb + y] *= 8.f / 7.f;
    }
}

/******************************************************************************/

// HOG helper: compute 2x2 block normalization values (padded by 1 pixel)
float* hogNormMatrix(float *H, int nOrients, int hb, int wb, int bin) {
    float *N, *N1, *n; int o, x, y, dx, dy, hb1 = hb + 1, wb1 = wb + 1;
    float eps = 1e-4f / 4 / bin / bin / bin / bin; // precise backward equality
    N = (float*)wrCalloc(hb1*wb1, sizeof(float)); N1 = N + hb1 + 1;
    for (o = 0; o<nOrients; o++) for (x = 0; x<wb; x++) for (y = 0; y<hb; y++)
        N1[x*hb1 + y] += H[o*wb*hb + x*hb + y] * H[o*wb*hb + x*hb + y];
    for (x = 0; x<wb - 1; x++) for (y = 0; y<hb - 1; y++) {
        n = N1 + x*hb1 + y; *n = 1 / float(sqrt(n[0] + n[1] + n[hb1] + n[hb1 + 1] + eps));
    }
    x = 0;     dx = 1; dy = 1; y = 0;                  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = 0;     dx = 1; dy = 0; for (y = 0; y<hb1; y++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = 0;     dx = 1; dy = -1; y = hb1 - 1;              N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = wb1 - 1; dx = -1; dy = 1; y = 0;                  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = wb1 - 1; dx = -1; dy = 0; for (y = 0; y<hb1; y++) N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = wb1 - 1; dx = -1; dy = -1; y = hb1 - 1;              N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    y = 0;     dx = 0; dy = 1; for (x = 0; x<wb1; x++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    y = hb1 - 1; dx = 0; dy = -1; for (x = 0; x<wb1; x++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    return N;
}

// HOG helper: compute HOG or FHOG channels
void hogChannels(float *H, const float *R, const float *N,
    int hb, int wb, int nOrients, float clip, int type)
{
#define GETT(blk) t=R1[y]*N1[y-(blk)]; if(t>clip) t=clip; c++;
    const float r = .2357f; int o, x, y, c; float t;
    const int nb = wb*hb, nbo = nOrients*nb, hb1 = hb + 1;
    for (o = 0; o<nOrients; o++) for (x = 0; x<wb; x++) {
        const float *R1 = R + o*nb + x*hb, *N1 = N + x*hb1 + hb1 + 1;
        float *H1 = (type <= 1) ? (H + o*nb + x*hb) : (H + x*hb);
        if (type == 0) for (y = 0; y<hb; y++) {
            // store each orientation and normalization (nOrients*4 channels)
            c = -1; GETT(0); H1[c*nbo + y] = t; GETT(1); H1[c*nbo + y] = t;
            GETT(hb1); H1[c*nbo + y] = t; GETT(hb1 + 1); H1[c*nbo + y] = t;
        }
        else if (type == 1) for (y = 0; y<hb; y++) {
            // sum across all normalizations (nOrients channels)
            c = -1; GETT(0); H1[y] += t*.5f; GETT(1); H1[y] += t*.5f;
            GETT(hb1); H1[y] += t*.5f; GETT(hb1 + 1); H1[y] += t*.5f;
        }
        else if (type == 2) for (y = 0; y<hb; y++) {
            // sum across all orientations (4 channels)
            c = -1; GETT(0); H1[c*nb + y] += t*r; GETT(1); H1[c*nb + y] += t*r;
            GETT(hb1); H1[c*nb + y] += t*r; GETT(hb1 + 1); H1[c*nb + y] += t*r;
        }
    }
#undef GETT
}

// compute HOG features
void hog(float *M, float *O, float *H, int h, int w, int binSize,
    int nOrients, int softBin, bool full, float clip)
{
    float *N, *R; const int hb = h / binSize, wb = w / binSize, nb = hb*wb;
    // compute unnormalized gradient histograms
    R = (float*)wrCalloc(wb*hb*nOrients, sizeof(float));
    gradHist(M, O, R, h, w, binSize, nOrients, softBin, full);
    // compute block normalization values
    N = hogNormMatrix(R, nOrients, hb, wb, binSize);
    // perform four normalizations per spatial block
    hogChannels(H, R, N, hb, wb, nOrients, clip, 0);
    wrFree(N); wrFree(R);
}

// compute FHOG features
void fhog(float *M, float *O, float *H, int h, int w, int binSize,
    int nOrients, int softBin, float clip)
{
    const int hb = h / binSize, wb = w / binSize, nb = hb*wb, nbo = nb*nOrients;
    float *N, *R1, *R2; int o, x;
    // compute unnormalized constrast sensitive histograms
    R1 = (float*)wrCalloc(wb*hb*nOrients * 2, sizeof(float));
    gradHist(M, O, R1, h, w, binSize, nOrients * 2, softBin, true);
    // compute unnormalized contrast insensitive histograms
    R2 = (float*)wrCalloc(wb*hb*nOrients, sizeof(float));
    for (o = 0; o<nOrients; o++) for (x = 0; x<nb; x++)
        R2[o*nb + x] = R1[o*nb + x] + R1[(o + nOrients)*nb + x];
    // compute block normalization values
    N = hogNormMatrix(R2, nOrients, hb, wb, binSize);
    // normalized histograms and texture channels
    hogChannels(H + nbo * 0, R1, N, hb, wb, nOrients * 2, clip, 1);
    hogChannels(H + nbo * 2, R2, N, hb, wb, nOrients * 1, clip, 1);
    hogChannels(H + nbo * 3, R1, N, hb, wb, nOrients * 2, clip, 2);
    wrFree(N); wrFree(R1); wrFree(R2);
}

cv::Mat mGradHist(cv::Mat img1, cv::Mat img2, int binSize, int nOrients, int softBin)
{
    int useHog = 0;
    float clipHog = 0.2f;
    bool full = false;
    int h = img1.rows;
    int w = img1.cols;
    int nChns = useHog == 0 ? nOrients : (useHog == 1 ? nOrients * 4 : nOrients * 3 + 5);
    int  hb = h / binSize, wb = w / binSize;
    if (2 == binSize)
    {
        hb = (h + 1) / binSize;
        wb = (w + 1) / binSize;
    }

    int num_filter_used = 4;
    int filtered_img_size[3] = { num_filter_used, wb, hb };
    cv::Mat resultImg = cv::Mat::zeros(3, filtered_img_size, CV_32F);

    if (nOrients == 0)
    {
        return resultImg;
    }
    cv::transpose(img1, img1);
    cv::transpose(img2, img2);

    if (useHog == 0)
    {
        gradHist((float *)img1.data, (float *)img2.data, (float *)resultImg.data, h, w, binSize, nOrients, softBin, full);
    }
    //cv::Mat tmp_result = cv::Mat::zeros(wb, hb, CV_32F);
    //std::memcpy(tmp_result.data, resultImg.data + 0*tmp_result.rows*tmp_result.cols*tmp_result.elemSize1(), tmp_result.rows*tmp_result.cols*tmp_result.elemSize1());
    //std::memcpy(tmp_result.data, resultImg.data, tmp_result.rows*tmp_result.cols*tmp_result.elemSize1());

    int filtered_img_size1[3] = { num_filter_used, hb, wb };
    cv::Mat result = cv::Mat::zeros(3, filtered_img_size1, CV_32F);

    // transpose 3d cv::Mat
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < resultImg.size[2]; j++)
        {
            for (size_t k = 0; k < resultImg.size[1]; k++)
            {
                result.at<float>(i,j,k) = resultImg.at<float>(i,k,j);
            }
        }

    }
    //cv::transpose(resultImg, result);

    return result;
}

cv::Mat mGradMag(cv::Mat img, int channel, int full)
{
    int c = channel;
    int h = img.rows;
    int w = img.cols;
    int d = 3;
    //if (c>0 && c <= d) { I += h*w*(c - 1); d = 1; }
    cv::Mat M(w, h, CV_32FC1);
    cv::Mat O(w, h, CV_32FC1);
    
    //transform input
    cv::Mat img_chan[3];
    cv::split(img, img_chan);

    // transform input img
    for (int i_ch = 0; i_ch < 3; ++i_ch)
        cv::transpose(img_chan[i_ch], img_chan[i_ch]);
    cv::Mat img_merge = cv::Mat(img.cols * 3, img.rows, CV_32F);
    std::memcpy(img_merge.data + 0 * img.cols * img.rows * img.elemSize1(),
        img_chan[2].data, img.cols * img.rows * img.elemSize1());
    std::memcpy(img_merge.data + 1 * img.cols * img.rows * img.elemSize1(),
        img_chan[1].data, img.cols * img.rows * img.elemSize1());
    std::memcpy(img_merge.data + 2 * img.cols * img.rows * img.elemSize1(),
        img_chan[0].data, img.cols * img.rows * img.elemSize1());
    img_merge = img_merge.reshape(1, 1);

    gradMag((float *)img_merge.data, (float *)M.data, (float *)O.data,  h,  w,  d, full>0);
    cv::transpose(M, M);
    cv::transpose(O, O);
    
    cv::Mat temp[2];
    temp[0] = M;
    temp[1] = O;
    cv::Mat result;
    cv::merge(temp, 2, result);

    return result;
}

cv::Mat mGradMagNorm(cv::Mat M, cv::Mat S, float mGradMagNorm)
{
    int h = M.rows;
    int w = M.cols;

    cv::transpose(M, M);
    cv::transpose(S, S);

    gradMagNorm((float *)M.data, (float *)S.data, h, w, mGradMagNorm);

    cv::transpose(M, M);
    return M;
}

void buildLookupSs(uint32 *&cids1, uint32 *&cids2, int *dims, int w, int m) 
{
    int i, j, z, z1, c, r; int locs[1024];
    int m2 = m*m, n = m2*(m2 - 1) / 2 * dims[2], s = int(w / m / 2.0 + .5);
    cids1 = new uint32[n]; cids2 = new uint32[n]; n = 0;
    for (i = 0; i<m; i++) locs[i] = uint32((i + 1)*(w + 2 * s - 1) / (m + 1.0) - s + .5);
    for (z = 0; z<dims[2]; z++) for (i = 0; i<m2; i++) for (j = i + 1; j<m2; j++) {
        z1 = z*dims[0] * dims[1]; n++;
        r = i%m; c = (i - r) / m; cids1[n - 1] = z1 + locs[c] * dims[0] + locs[r];
        r = j%m; c = (j - r) / m; cids2[n - 1] = z1 + locs[c] * dims[0] + locs[r];
    }
}

uint32* buildLookup(int *dims, int w) 
{
    int c, r, z, n = w*w*dims[2]; uint32 *cids = new uint32[n]; n = 0;
    for (z = 0; z<dims[2]; z++) for (c = 0; c<w; c++) for (r = 0; r<w; r++)
        cids[n++] = z*dims[0] * dims[1] + c*dims[0] + r;
    return cids;
}

template<typename T> inline T min(T x, T y) 
{ return x<y ? x : y; }

cv::Mat EdgeDetectMex(bici2::EdgeModel model, cv::Mat img, cv::Mat chns, cv::Mat chnsSs)
{
    std::vector<float> chns_vect;
    std::vector<float> chnsSs_vect;

    // transform 13 dims cv::Mat into 1D vector, cols first
    for (size_t i = 0; i < chns.size[0]; i++)
    {
        for (size_t j = 0; j < chns.size[2]; j++)
        {
            for (size_t k = 0; k < chns.size[1]; k++)
            {
                chns_vect.push_back(chns.at<float>(i,k,j));
            }
        }
    }
    for (size_t i = 0; i < chnsSs.size[0]; i++)
    {
        for (size_t j = 0; j < chnsSs.size[2]; j++)
        {
            for (size_t k = 0; k < chnsSs.size[1]; k++)
            {
                chnsSs_vect.push_back(chnsSs.at<float>(i,k,j));
            }
        }
    }
    
    std::vector<int> imgSize;
    imgSize.push_back(img.rows);
    imgSize.push_back(img.cols);
    int h = imgSize[0];
    int w = imgSize[1];
    int z = (img.dims <= 2) ? 1 : img.size[0];
    int nTreeNodes = 270751;
    int nTrees = 8;
    int h1 = std::ceil((double)(h - model.opts.imWidth) / model.opts.stride);
    int w1 = std::ceil((double)(w - model.opts.imWidth) / model.opts.stride);
    int h2 = h1*model.opts.stride + model.opts.gtWidth;
    int w2 = w1*model.opts.stride + model.opts.gtWidth;
    //std::vector<int> imgDims = { h, w, z };
    int imgDims[3] = { h, w, z };
    //std::vector<int> chnDims= { h / model.opts.shrink, w / model.opts.shrink, model.opts.nChns };
    //std::vector<int> indDims = { h1, w1, model.opts.nTreesEval };
    //std::vector<int> outDims = { h2, w2, 1 };
    //std::vector<int> segDims = { model.opts.gtWidth, model.opts.gtWidth, h1, w1, model.opts.nTreesEval };
    int chnDims[3] = { h / model.opts.shrink, w / model.opts.shrink, model.opts.nChns };
    int indDims[3] = { h1, w1, model.opts.nTreesEval };
    int outDims[3] = { h2, w2, 1 };
    int segDims[5] = { model.opts.gtWidth, model.opts.gtWidth, h1, w1, model.opts.nTreesEval };

    uint32 *iids, *eids, *cids, *cids1, *cids2;
    iids = bici2::buildLookup((int*)imgDims, model.opts.gtWidth);
    eids = bici2::buildLookup((int*)outDims, model.opts.gtWidth);
    cids = bici2::buildLookup((int*)chnDims, model.opts.imWidth / model.opts.shrink);
    buildLookupSs(cids1, cids2, (int*)chnDims, model.opts.imWidth / model.opts.shrink, model.opts.nCells);
    
    //create output
    std::vector<unsigned long int> ind(indDims[0] * indDims[1] * indDims[2]);


    //std::vector<unsigned int> E;
    cv::Mat E = cv::Mat::zeros(outDims[0], outDims[1], CV_32FC1);
    cv::transpose(E, E);
    std::vector<unsigned long int> segsOut;

    int nThreads = model.opts.nThreads;
    #ifdef USEOMP
    nThreads = std::min(nThreads, omp_get_max_threads());
    #pragma omp parallel for num_threads(nThreads)
    #endif

    for (int c = 0; c<w1; c++) 
    {
        for (int t = 0; t < model.opts.nTreesEval; t++)
        {
                for (int r0 = 0; r0 < 2; r0++) for (int r = r0; r < h1; r += 2)
                {
                    int o = (r*model.opts.stride / model.opts.shrink) + (c*model.opts.stride / model.opts.shrink)*h / model.opts.shrink;
                    // select tree to evaluate
                    int t1 = ((r + c) % 2 * model.opts.nTreesEval + t) % nTrees; uint32 k = t1*nTreeNodes;
                    while (model.child[k])
                    {
                        // compute feature (either channel or self-similarity feature)
                        uint32 f = model.fids[k];
                        float ftr;
                        if (f < model.opts.nChnFtrs) {
                            ftr = chns_vect[cids[f] + o];
                        }
                        else{
                            ftr = chnsSs_vect[cids1[f - model.opts.nChnFtrs] + o] - chnsSs_vect[cids2[f - model.opts.nChnFtrs] + o];
                        }
                        // compare ftr to threshold and move left or right accordingly
                        if (ftr < model.thrs[k]) {
                            k = model.child[k] - 1;
                        }
                        else {
                            k = model.child[k];
                        }
                        k += t1*nTreeNodes;
                    }
                    // store leaf index and update edge maps
                    ind[r + c*h1 + t*h1*w1] = k;
                }
        }
    }
    int nBnds = (model.eBnds.size() - 1) / model.thrs.size();
    for (int c0 = 0; c0<model.opts.gtWidth / model.opts.stride; c0++) 
    {
    #ifdef USEOMP
    #pragma omp parallel for num_threads(nThreads)
    #endif
        //sharpen = 0;
        for (int c = c0; c<w1; c += model.opts.gtWidth / model.opts.stride) 
        {
            for (int r = 0; r<h1; r++) for (int t = 0; t<model.opts.nTreesEval; t++) 
            {
                uint32 k = ind[r + c*h1 + t*h1*w1];
                float *E1 = (float *)E.data + (r*model.opts.stride) + (c*model.opts.stride)*h2;
                int b0 = model.eBnds[k*nBnds], b1 = model.eBnds[k*nBnds + 1]; if (b0 == b1) continue;
                for (int b = b0; b<b1; b++) E1[eids[model.eBins[b]]]++;
            }
        }
    }
    //still have question about E1 and E;
    delete[] iids; delete[] eids;
    delete[] cids; delete[] cids1; delete[] cids2;

    cv::transpose(E, E);
    return E;
}



}


