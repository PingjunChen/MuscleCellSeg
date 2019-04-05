/* Pingjun Adapted from  Pablo Arbelaez */

#include "ucm_mean_pb.h"

namespace bici2
{
    void Region::merge(Region& r, int* labels, const int& label, double* ucm, 
                        const double& saliency, const int& son, const int& tx)
    {
        /* 			I. BOUNDARY        */

        // 	Ia. update father's boundary
        std::list<Bdry_element>::iterator itrb, itrb2;
        itrb = boundary.begin();
        while (itrb != boundary.end())
        {
            if (labels[(*itrb).cc_neigh] == son)
            {
                itrb2 = itrb;
                ++itrb;
                boundary.erase(itrb2);
            }
            else ++itrb;
        }

        int coord_contour;

        //	Ib. move son's boundary to father
        for (itrb = r.boundary.begin(); itrb != r.boundary.end(); ++itrb)
        {
            if (ucm[(*itrb).coord] < saliency) ucm[(*itrb).coord] = saliency;

            if (labels[(*itrb).cc_neigh] != label)
                boundary.push_back(Bdry_element(*itrb));

        }
        r.boundary.erase(r.boundary.begin(), r.boundary.end());

        /* 			II. ELEMENTS      */

        for (std::list<int>::iterator p = r.elements.begin(); p != r.elements.end(); ++p) labels[*p] = label;
        elements.insert(elements.begin(), r.elements.begin(), r.elements.end());
        r.elements.erase(r.elements.begin(), r.elements.end());


        /* 			III. NEIGHBORS        */

        std::map<int, Neighbor_Region, std::less<int> >::iterator itr, itr2;

        // 	IIIa. remove inactive neighbors from father
        itr = neighbors.begin();
        while (itr != neighbors.end())
        {
            if (labels[(*itr).first] != (*itr).first)
            {
                itr2 = itr;
                ++itr;
                neighbors.erase(itr2);
            }
            else ++itr;
        }

        // 	IIIb. remove inactive neighbors from son y and neighbors belonging to father
        itr = r.neighbors.begin();
        while (itr != r.neighbors.end())
        {
            if ((labels[(*itr).first] != (*itr).first) || (labels[(*itr).first] == label))
            {
                itr2 = itr;
                ++itr;
                r.neighbors.erase(itr2);
            }
            else ++itr;
        }
    }

    void complete_contour_map(double* ucm, const int& txc, const int& tyc)
        /* complete contour map by max strategy on Khalimsky space  */
    {
        int vx[4] = { 1, 0, -1, 0 };
        int vy[4] = { 0, 1, 0, -1 };
        int nxp, nyp, cv;
        double maximo;

        for (int x = 0; x < txc; x = x + 2) for (int y = 0; y < tyc; y = y + 2)
        {
            maximo = 0.0;
            for (int v = 0; v < 4; v++)
            {
                nxp = x + vx[v]; nyp = y + vy[v]; cv = nxp + nyp * txc;
                if ((nyp >= 0) && (nyp < tyc) && (nxp < txc) && (nxp >= 0) && (maximo < ucm[cv]))
                    maximo = ucm[cv];
            }
            ucm[x + y*txc] = maximo;
        }

    }

    void compute_ucm(double* local_boundaries, int* initial_partition, const int& totcc, 
                        double* ucm, const int& tx, const int& ty)
    {
        // I. INITIATE
        int p, c;
        int* labels = new int[totcc];

        for (c = 0; c < totcc; c++)
        {
            labels[c] = c;
        }

        // II. ULTRAMETRIC
        Region* R = new Region[totcc];
        std::priority_queue<Order_node, std::vector<Order_node>, std::less<Order_node> > merging_queue;
        double totalPb, totalBdry, dissimilarity;
        int v, px;

        for (p = 0; p < (2 * tx + 1)*(2 * ty + 1); p++) ucm[p] = 0.0;

        // INITIATE REGI0NS
        for (c = 0; c < totcc; c++) R[c] = Region(c);

        // INITIATE UCM
        int vx[4] = { 1, 0, -1, 0 };
        int vy[4] = { 0, 1, 0, -1 };
        int nxp, nyp, cnp, xp, yp, label;

        for (p = 0; p < tx*ty; p++)
        {
            xp = p%tx; yp = p / tx;
            for (v = 0; v < 4; v++)
            {
                nxp = xp + vx[v]; nyp = yp + vy[v]; cnp = nxp + nyp*tx;
                if ((nyp >= 0) && (nyp < ty) && (nxp < tx) && (nxp >= 0) && (initial_partition[cnp] != initial_partition[p]))
                    R[initial_partition[p]].boundary.push_back(Bdry_element((xp + nxp + 1) + (yp + nyp + 1)*(2 * tx + 1), initial_partition[cnp]));
            }
        }

        // INITIATE merging_queue
        std::list<Bdry_element>::iterator itrb;
        for (c = 0; c < totcc; c++)
        {
            R[c].boundary.sort();

            label = (*R[c].boundary.begin()).cc_neigh;
            totalBdry = 0.0;
            totalPb = 0.0;

            for (itrb = R[c].boundary.begin(); itrb != R[c].boundary.end(); ++itrb)
            {
                if ((*itrb).cc_neigh == label)
                {
                    totalBdry++;
                    totalPb += local_boundaries[(*itrb).coord];
                }
                else
                {
                    R[c].neighbors[label] = Neighbor_Region(totalPb / totalBdry, totalPb, totalBdry);
                    if (label > c)   merging_queue.push(Order_node(totalPb / totalBdry, c, label));
                    label = (*itrb).cc_neigh;
                    totalBdry = 1.0;
                    totalPb = local_boundaries[(*itrb).coord];
                }

            }
            R[c].neighbors[label] = Neighbor_Region(totalPb / totalBdry, totalPb, totalBdry);
            if (label > c)   merging_queue.push(Order_node(totalPb / totalBdry, c, label));
        }


        //MERGING
        Order_node minor;
        int father, son;
        std::map<int, Neighbor_Region, std::less<int> >::iterator itr;
        double current_energy = 0.0;

        while (!merging_queue.empty())
        {
            minor = merging_queue.top(); merging_queue.pop();
            if ((labels[minor.region1] == minor.region1) && (labels[minor.region2] == minor.region2) &&
                (minor.energy == R[minor.region1].neighbors[minor.region2].energy))
            {
                if (current_energy <= minor.energy) current_energy = minor.energy;
                else
                {
                    printf("\n ERROR : \n");
                    printf("\n current_energy = %f \n", current_energy);
                    printf("\n minor.energy = %f \n\n", minor.energy);
                    delete[] R; delete[] labels;
                    // mexErrMsgTxt(" BUG: THIS IS NOT AN ULTRAMETRIC !!! ");
                }

                dissimilarity = R[minor.region1].neighbors[minor.region2].total_pb / R[minor.region1].neighbors[minor.region2].bdry_length;

                if (minor.region1 < minor.region2)
                {
                    son = minor.region1; father = minor.region2;
                }
                else
                {
                    son = minor.region2; father = minor.region1;
                }

                R[father].merge(R[son], labels, father, ucm, dissimilarity, son, tx);

                // move and update neighbors
                while (R[son].neighbors.size() > 0)
                {
                    itr = R[son].neighbors.begin();

                    R[father].neighbors[(*itr).first].total_pb += (*itr).second.total_pb;
                    R[(*itr).first].neighbors[father].total_pb += (*itr).second.total_pb;

                    R[father].neighbors[(*itr).first].bdry_length += (*itr).second.bdry_length;
                    R[(*itr).first].neighbors[father].bdry_length += (*itr).second.bdry_length;

                    R[son].neighbors.erase(itr);
                }

                // update merging_queue
                for (itr = R[father].neighbors.begin(); itr != R[father].neighbors.end(); ++itr)
                {

                    dissimilarity = R[father].neighbors[(*itr).first].total_pb / R[father].neighbors[(*itr).first].bdry_length;

                    merging_queue.push(Order_node(dissimilarity, (*itr).first, father));
                    R[father].neighbors[(*itr).first].energy = dissimilarity;
                    R[(*itr).first].neighbors[father].energy = dissimilarity;

                }
            }
        }

        complete_contour_map(ucm, 2 * tx + 1, 2 * ty + 1);

        delete[] R; 
        delete[] labels;
    }
}