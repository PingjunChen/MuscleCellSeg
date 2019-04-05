/* Pingjun Adapted from  Pablo Arbelaez */

#ifndef MUSCLEMINER_UCM_MEAN_PB_H_
#define MUSCLEMINER_UCM_MEAN_PB_H_

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <deque>
#include <queue>
#include <vector>
#include <list>
#include <map>

#include "export.h"

namespace bici2
{
    class MUSCLEMINER_EXPORT Order_node
    {
    public:
        double energy;
        int region1;
        int region2;

        Order_node(){ energy = 0.0; region1 = 0; region2 = 0; }

        Order_node(const double& e, const int& rregion1, const int& rregion2)
        {
            energy = e;
            region1 = rregion1;
            region2 = rregion2;
        }

        ~Order_node(){}
        // LEXICOGRAPHIC ORDER on priority queue: (energy,label)
        bool operator < (const Order_node& x) const
        {
            return ((energy > x.energy)
                || ((energy == x.energy) && (region1 > x.region1))
                || ((energy == x.energy) && (region1 == x.region1) && (region2 > x.region2))
                );
        }
    };

    class MUSCLEMINER_EXPORT Neighbor_Region
    {
    public:
        double energy;
        double total_pb;
        double bdry_length;

        Neighbor_Region()
        {
            energy = 0.0; total_pb = 0.0; bdry_length = 0.0;
        }

        Neighbor_Region(const Neighbor_Region& v)
        {
            energy = v.energy; total_pb = v.total_pb; bdry_length = v.bdry_length;
        }

        Neighbor_Region(const double& en, const double& tt, const double& bor)
        {
            energy = en;
            total_pb = tt;
            bdry_length = bor;
        }

        ~Neighbor_Region(){}
    };

    class MUSCLEMINER_EXPORT Bdry_element
    {
    public:
        int coord;
        int cc_neigh;

        Bdry_element(){}

        Bdry_element(const int& c, const int& v) { coord = c; cc_neigh = v; }

        Bdry_element(const Bdry_element& n) { coord = n.coord; cc_neigh = n.cc_neigh; }

        ~Bdry_element(){}

        bool operator ==(const Bdry_element& n) const 
        { 
            return ((coord == n.coord) && (cc_neigh == n.cc_neigh)); 
        }

        // LEXICOGRAPHIC ORDER: (cc_neigh, coord)
        bool operator < (const Bdry_element& n) const 
        { 
            return ((cc_neigh < n.cc_neigh) 
                    || ((cc_neigh == n.cc_neigh) && (coord < n.coord))); 
        }
    };

    class MUSCLEMINER_EXPORT Region
    {
    public:
        std::list<int> elements;
        std::map<int, Neighbor_Region, std::less<int> > neighbors;
        std::list<Bdry_element> boundary;

        Region(){}

        Region(const int& l) { elements.push_back(l); }

        ~Region(){}

        void merge(Region& r, int* labels, const int& label, double* ucm, 
                    const double& saliency, const int& son, const int& tx);
    };

    /* complete contour map by max strategy on Khalimsky space  */
    MUSCLEMINER_EXPORT void complete_contour_map
        (double* ucm, const int& txc, const int& tyc);

    /*************************************************************************/
    MUSCLEMINER_EXPORT void compute_ucm
        (double* local_boundaries, int* initial_partition, const int& totcc, 
         double* ucm, const int& tx, const int& ty);
}

#endif //MUSCLEMINER_UCM_MEAN_PB_H_