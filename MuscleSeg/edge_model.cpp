#include "edge_model.h"

namespace bici2
{
    void EdgeModel::ParseOpts(const std::string& path)
    {
        std::ifstream ifs(path);
        std::string content;
        std::getline(ifs, content);

        while (!ifs.eof())
        {
            std::getline(ifs, content);
            if (content.substr(0, 2) == "##")
            {
                std::string field_name = content.substr(2);
                int field_line_num = 0;
                if (field_name == "opts")
                {
                    struct EdgeModelOpts opts;
                    field_line_num = 37;
                    std::size_t found = -1;
                    std::string opt_field_name = "";
                    for (int i_field = 0; i_field < field_line_num; ++i_field)
                    {
                        std::getline(ifs, content);
                        found = content.find_first_of(" ");
                        opt_field_name = content.substr(0, found);
                        if (opt_field_name == "imWidth")
                            opts.imWidth = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "gtWidth")
                            opts.gtWidth = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nEdgeBins")
                            opts.nEdgeBins = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nPos")
                            opts.nPos = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nNeg")
                            opts.nNeg = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nImgs")
                            //opts.nImgs = -1.0; // Inf (in fact)
                            opts.nImgs = std::numeric_limits<int>::max();
                        else if (opt_field_name == "nTrees")
                            opts.nTrees = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "fracFtrs")
                            opts.fracFtrs = std::stof(content.substr(found + 1));
                        else if (opt_field_name == "minCount")
                            opts.minCount = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "minChild")
                            opts.minChild = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "maxDepth")
                            opts.maxDepth = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "discretize")
                            opts.discretize = content.substr(found + 1);
                        else if (opt_field_name == "nSamples")
                            opts.nSamples = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nClasses")
                            opts.nClasses = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "split")
                            opts.split = content.substr(found + 1);
                        else if (opt_field_name == "nOrients")
                            opts.nOrients = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "grdSmooth")
                            opts.grdSmooth = std::stof(content.substr(found + 1));
                        else if (opt_field_name == "chnSmooth")
                            opts.chnSmooth = std::stof(content.substr(found + 1));
                        else if (opt_field_name == "simSmooth")
                            opts.simSmooth = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "normRad")
                            opts.normRad = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "shrink")
                            opts.shrink = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nCells")
                            opts.nCells = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "rgbd")
                            opts.rgbd = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "stride")
                            opts.stride = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "multiscale")
                            opts.multiscale = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nTreesEval")
                            opts.nTreesEval = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nThreads")
                            opts.nThreads = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nms")
                            opts.nms = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "seed")
                            opts.seed = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "useParfor")
                            opts.useParfor = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "modelDir")
                            opts.modelDir = content.substr(found + 1);
                        else if (opt_field_name == "modelFnm")
                            opts.modelFnm = content.substr(found + 1);
                        else if (opt_field_name == "bsdsDir")
                            opts.bsdsDir = content.substr(found + 1);
                        else if (opt_field_name == "nChns")
                            opts.nChns = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nChnFtrs")
                            opts.nChnFtrs = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nSimFtrs")
                            opts.nSimFtrs = std::stoi(content.substr(found + 1));
                        else if (opt_field_name == "nTotFtrs")
                            opts.nTotFtrs = std::stoi(content.substr(found + 1));
                    }
                    this->opts = opts;
                    //break;
                }
            }
        }
        ifs.close();
    }

    void EdgeModel::ParseThrs(const std::string& path)
    {
        std::ifstream ifs(path.c_str(), std::ios::binary);
        ifs.seekg(0, ifs.end);     //N is the total number of doubles
        long int array_len = ifs.tellg() / sizeof(float);
        ifs.seekg(0, ifs.beg);

        thrs.resize(array_len);
        ifs.read(reinterpret_cast<char*>(thrs.data()), thrs.size()*sizeof(float));
    }

    void EdgeModel::ParseFids(const std::string& path)
    {
        std::ifstream ifs(path.c_str(), std::ios::binary);
        ifs.seekg(0, ifs.end);     //N is the total number of doubles
        long int array_len = ifs.tellg() / sizeof(unsigned int);
        ifs.seekg(0, ifs.beg);

        fids.resize(array_len);
        ifs.read(reinterpret_cast<char*>(fids.data()), fids.size()*sizeof(unsigned int));
    }

    void EdgeModel::ParseChild(const std::string& path)
    {
        std::ifstream ifs(path.c_str(), std::ios::binary);
        ifs.seekg(0, ifs.end);     //N is the total number of doubles
        long int array_len = ifs.tellg() / sizeof(unsigned int);
        ifs.seekg(0, ifs.beg);

        child.resize(array_len);
        ifs.read(reinterpret_cast<char*>(child.data()), child.size()*sizeof(unsigned int));
    }

    void EdgeModel::ParseCount(const std::string& path)
    {
        std::ifstream ifs(path.c_str(), std::ios::binary);
        ifs.seekg(0, ifs.end);     //N is the total number of doubles
        long int array_len = ifs.tellg() / sizeof(unsigned int);
        ifs.seekg(0, ifs.beg);

        count.resize(array_len);
        ifs.read(reinterpret_cast<char*>(count.data()), count.size()*sizeof(unsigned int));
    }

    void EdgeModel::ParseDepth(const std::string& path)
    {
        std::ifstream ifs(path.c_str(), std::ios::binary);
        ifs.seekg(0, ifs.end);     //N is the total number of doubles
        long int array_len = ifs.tellg() / sizeof(unsigned int);
        ifs.seekg(0, ifs.beg);

        depth.resize(array_len);
        ifs.read(reinterpret_cast<char*>(depth.data()), depth.size()*sizeof(unsigned int));
    }


    void EdgeModel::ParseEbins(const std::string& path)
    {
        std::ifstream ifs(path.c_str(), std::ios::binary);
        ifs.seekg(0, ifs.end);     //N is the total number of doubles
        long int array_len = ifs.tellg() / sizeof(unsigned short);
        ifs.seekg(0, ifs.beg);

        eBins.resize(array_len);
        ifs.read(reinterpret_cast<char*>(eBins.data()), eBins.size()*sizeof(unsigned short));
    }


    void EdgeModel::ParseEbnds(const std::string& path)
    {
        std::ifstream ifs(path.c_str(), std::ios::binary);
        ifs.seekg(0, ifs.end);     //N is the total number of doubles
        long int array_len = ifs.tellg() / sizeof(unsigned int);
        ifs.seekg(0, ifs.beg);

        eBnds.resize(array_len);
        ifs.read(reinterpret_cast<char*>(eBnds.data()), eBnds.size()*sizeof(unsigned int));
    }

    void EdgeModel::ParseEdgeModel(const std::string& opts_path,
        const std::string& thrs_path,
        const std::string& fids_path,
        const std::string& child_path,
        const std::string& count_path,
        const std::string& depth_path,
        const std::string& ebins_path,
        const std::string& ebnds_path)
    {
        this->ParseOpts(opts_path);
        this->ParseThrs(thrs_path);
        this->ParseFids(fids_path);
        this->ParseChild(child_path);
        this->ParseCount(count_path);
        this->ParseDepth(depth_path);
        this->ParseEbins(ebins_path);
        this->ParseEbnds(ebnds_path);
    }
}
