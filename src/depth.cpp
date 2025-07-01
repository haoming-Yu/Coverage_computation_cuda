#include "depth.h"
#include <iostream>
#include <fstream>  
#include <sys/types.h>
#include <dirent.h>

namespace Depth {

Depth::Depth() {}

// load depth maps from the given folder
void Depth::loadDepthMaps(const std::string& depth_map_file_folder) {
    std::vector<std::string> depth_map_files;
    // read all the files in the folder
    DIR* dir = opendir(depth_map_file_folder.c_str());
    if (dir == nullptr) {
        std::cerr << "Failed to open directory: " << depth_map_file_folder << std::endl;
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            depth_map_files.push_back(entry->d_name);
        }
    }
    depth_map_num_ = depth_map_files.size();
    std::cout << "Loaded " << depth_map_num_ << " depth maps" << std::endl;

    // sort the depth map files by the number at the end of the file name
    std::sort(depth_map_files.begin(), depth_map_files.end(), [](const std::string& a, const std::string& b) {
        return std::stoi(a.substr(a.find_last_of('_') + 1)) < std::stoi(b.substr(b.find_last_of('_') + 1));
    });

    // for debugging, check the depth image's sequence, checked, correct

    // load the depth maps
    for (const auto& file : depth_map_files) {
        cv::Mat depth_map = cv::imread(depth_map_file_folder + "/" + file, cv::IMREAD_UNCHANGED);
        depth_maps_.push_back(depth_map);
    }

    std::cout << "Depth map size: " << depth_maps_[0].size() << std::endl;
    std::cout << "first depth map's first pixel value (depth in millimeter actually, thus 1000 is the depth scale): " << depth_maps_[0].at<unsigned short>(0, 0) << std::endl;
}

// convert the depth maps to float, at the sequence of: depth_maps_sequence, rows, cols
// this method returns the original resolution's depth map
float* Depth::convertToFloat() {
    if (this->depth_map_num_ == 0) {
        return nullptr;
    }
    // convert the depth maps to float, must ensure that there are at least one depth map
    // the sequence should be: depth map, rows, cols
    this->depth_maps_float_ = new float[this->depth_map_num_ * this->depth_maps_[0].rows * this->depth_maps_[0].cols];
    for (int i = 0; i < this->depth_map_num_; i++) {
        for (int j = 0; j < this->depth_maps_[0].rows; j++) {
            for (int k = 0; k < this->depth_maps_[0].cols; k++) {
                this->depth_maps_float_[i * this->depth_maps_[0].rows * this->depth_maps_[0].cols + j * this->depth_maps_[0].cols + k] = this->depth_maps_[i].at<unsigned short>(j, k) / 1000.0f;
            }
        }
    }
    return this->depth_maps_float_;
}

} // namespace Depth