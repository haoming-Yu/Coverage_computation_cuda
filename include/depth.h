#ifndef DEPTH_H
#define DEPTH_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace Depth {

/**
 * @brief class to read-in the depth map from the depth map file.
 * 
 * Here our depth shares the same intrinsic and extrinsics to the camera with same index.
 * Cause our depth map is actually rendered from mesh using rasterization by OpenGL.
 * 
 * Thus the Intrinsic and Extrinsic is not modeled here in depth map.
 * If you have the need to process depth maps separately, please refer to the camera.h and depth.h, and implement your own class~
 * 
 * @author: Haoming Yu
 * @date: 2025-06-30
 * @version: 1.0
 * @note: This class is not tested yet, please use it with caution. For Graphics's convenience, I set all the members to public.
 *  Only used for reading depth map in, and convert to CUDA friendly format.
 */
class Depth {
public:
    Depth();
    ~Depth() {};
    void loadDepthMaps(const std::string& depth_map_file_folder); // load all the depth maps from the given folder
    float* convertToFloat();

public:
    std::vector<cv::Mat> depth_maps_;
    int depth_map_num_; // number of depth maps, also number of cameras
    float* depth_maps_float_;
};

} // namespace Depth

#endif // DEPTH_H