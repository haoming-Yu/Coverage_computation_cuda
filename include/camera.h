#ifndef CAM_H
#define CAM_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "cudaUtils.cuh"
#include <cuda_runtime.h>

/* Note that here the intrinsic is directly written in the given path, the format is:
 * fx = 1302.39546
 * fy = 1301.80638
 * cx = 689.52858
 * cy = 504.24215
 * width = 1280
 * height = 1024
 * 
 * The extrinsic is written in the given path, the format follows open3d colormap format.
 * The format is:
 * 0 0 1
 * R01 R02 R03 T0
 * R11 R12 R13 T1
 * R21 R22 R23 T2
 * 0 0 0 1
 * 1 1 2
 * ... (omitted)
 * 
 * If you need to use your own data, please modify the loadIntrinsic and loadExtrinsics functions or the format of the data.
 */

namespace Camera {

struct Intrinsic {
    float fx, fy, cx, cy;
    int img_width, img_height;
};

struct Extrinsic {
    Eigen::Matrix4d T_wc; // camera to world
    Eigen::Matrix3d R_wc; // rotation matrix
    Eigen::Vector3d t_wc; // translation vector

    Eigen::Matrix4d T_cw; // world to camera
    Eigen::Matrix3d R_cw; // rotation matrix
    Eigen::Vector3d t_cw; // translation vector

    Extrinsic() : T_wc(Eigen::Matrix4d::Identity()), R_wc(Eigen::Matrix3d::Identity()), t_wc(Eigen::Vector3d::Zero()), T_cw(Eigen::Matrix4d::Identity()), R_cw(Eigen::Matrix3d::Identity()), t_cw(Eigen::Vector3d::Zero()) {} // default to identity rotation and zero translation
};

struct Extrinsic_jetson {
    float t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23; // 3 * 4 matrix, the last row 0, 0, 0, 1 is not stored
};

class Cam {
public:
    Cam();
    ~Cam() {};
    void loadIntrinsic(const std::string& intrinsic_file);
    void loadIntrinsic_jetson(const std::string& intrinsic_file);
    void loadExtrinsics_colormap(const std::string& extrinsic_file); // this method loads the traj.log file format which adopted by COLORMAP
    void loadExtrinsics_colormap_jetson(const std::string& extrinsic_file); // use this method to support unified memory in jetson.

    Extrinsic getExtrinsic(int idx) { return extrinsics_[idx]; }
    Intrinsic getIntrinsic() { return intrinsic_; }
    float* dump_intrinsic_to_float() {
        this->float_intrinsic_ = new float[6];
        this->float_intrinsic_[0] = this->intrinsic_.fx;
        this->float_intrinsic_[1] = this->intrinsic_.fy;
        this->float_intrinsic_[2] = this->intrinsic_.cx;
        this->float_intrinsic_[3] = this->intrinsic_.cy;
        this->float_intrinsic_[4] = (float)this->intrinsic_.img_width;
        this->float_intrinsic_[5] = (float)this->intrinsic_.img_height;
        return this->float_intrinsic_;
    }
    float* dump_extrinsic_to_float() {
        this->float_extrinsic_ = new float[12 * this->extrinsics_.size()];
        for (int i = 0; i < this->extrinsics_.size(); i++) {
            this->float_extrinsic_[i * 12 + 0] = this->extrinsics_[i].T_cw(0, 0);
            this->float_extrinsic_[i * 12 + 1] = this->extrinsics_[i].T_cw(0, 1);
            this->float_extrinsic_[i * 12 + 2] = this->extrinsics_[i].T_cw(0, 2);
            this->float_extrinsic_[i * 12 + 3] = this->extrinsics_[i].T_cw(0, 3);
            this->float_extrinsic_[i * 12 + 4] = this->extrinsics_[i].T_cw(1, 0);
            this->float_extrinsic_[i * 12 + 5] = this->extrinsics_[i].T_cw(1, 1);
            this->float_extrinsic_[i * 12 + 6] = this->extrinsics_[i].T_cw(1, 2);
            this->float_extrinsic_[i * 12 + 7] = this->extrinsics_[i].T_cw(1, 3);
            this->float_extrinsic_[i * 12 + 8] = this->extrinsics_[i].T_cw(2, 0);
            this->float_extrinsic_[i * 12 + 9] = this->extrinsics_[i].T_cw(2, 1);
            this->float_extrinsic_[i * 12 + 10] = this->extrinsics_[i].T_cw(2, 2);
            this->float_extrinsic_[i * 12 + 11] = this->extrinsics_[i].T_cw(2, 3);
        }
        return this->float_extrinsic_;
    }

    // for images, we need to save space on memory, thus do not store images in class, just read out and directly save to a new folder
    bool filter_images_with_mask(unsigned int* mask, const std::string& image_src_folder, const std::string& image_dst_folder) {
        int num_camera_no_filter = this->number_extrinsics;
        for (int i = 0; i < num_camera_no_filter; i++) {
            if (mask[i] == 1) {
                std::string image_path = image_src_folder + "/imgs_" + std::to_string(i + 1) + ".jpg";
                cv::Mat image = cv::imread(image_path);
                if (image.empty()) {
                    std::cout << "Could not open image file: " << image_path << std::endl;
                    return false;
                }
                // here we keep the image name unchanged to debug easier
                std::string image_dst_path = image_dst_folder + "/imgs_" + std::to_string(i + 1) + ".jpg";
                cv::imwrite(image_dst_path, image);
                // std::cout << "Filtered image saved to: " << image_dst_path << std::endl;
            }
        }
        return true;
    }
    // no need for source, we've saved one extrinsic vector for cuda processing
    // and the extrinsic is saved in a format that is required by mvs-texture
    bool filter_extrinsics_with_mask(unsigned int* mask, const std::string& extrinsic_dst_folder) {
        int num_camera_no_filter = this->number_extrinsics;
        for (int i = 0; i < num_camera_no_filter; i++) {
            if (mask[i] == 1) {
                // here we name the corresponding extrinsic file the same name as the image file except the extension for mvs-texture to find.
                std::string extrinsic_path = extrinsic_dst_folder + "/imgs_" + std::to_string(i + 1) + ".cam";
                std::ofstream extrinsic_file(extrinsic_path);
                if (!extrinsic_file.is_open()) {
                    std::cout << "Could not open extrinsic file: " << extrinsic_path << std::endl;
                    return false;
                }
                // world to camera extrinsic
                // first row: tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22
                Eigen::Matrix4d T_cw;
                T_cw << this->unified_float_extrinsic_[i * 12 + 0], this->unified_float_extrinsic_[i * 12 + 1], this->unified_float_extrinsic_[i * 12 + 2], this->unified_float_extrinsic_[i * 12 + 3],
                        this->unified_float_extrinsic_[i * 12 + 4], this->unified_float_extrinsic_[i * 12 + 5], this->unified_float_extrinsic_[i * 12 + 6], this->unified_float_extrinsic_[i * 12 + 7],
                        this->unified_float_extrinsic_[i * 12 + 8], this->unified_float_extrinsic_[i * 12 + 9], this->unified_float_extrinsic_[i * 12 + 10], this->unified_float_extrinsic_[i * 12 + 11],
                        0, 0, 0, 1;

                Eigen::Matrix3d R_cw = T_cw.block<3, 3>(0, 0);
                Eigen::Vector3d t_cw = T_cw.block<3, 1>(0, 3);
                extrinsic_file << t_cw(0) << " " << t_cw(1) << " " << t_cw(2) << " " << R_cw(0, 0) << " " << R_cw(0, 1) << " " << R_cw(0, 2) << " " << R_cw(1, 0) << " " << R_cw(1, 1) << " " << R_cw(1, 2) << " " << R_cw(2, 0) << " " << R_cw(2, 1) << " " << R_cw(2, 2) << std::endl;
                // second row: f d0 d1 paspect ppx ppy
                float fx = this->unified_float_intrinsic_[0];
                float fy = this->unified_float_intrinsic_[1];
                float cx = this->unified_float_intrinsic_[2];
                float cy = this->unified_float_intrinsic_[3];
                float img_width = this->unified_float_intrinsic_[4];
                float img_height = this->unified_float_intrinsic_[5];
                extrinsic_file << fx / img_width << " " << 0.0 << " " << 0.0 << " " << fy / fx << " " << cx / img_width << " " << cy / img_height << std::endl;
                extrinsic_file.close();
            }
        }
        return true;
    }
public:
    float* float_intrinsic_;
    float* float_extrinsic_; // 3 * 4 world to camera extrinsics, to save space, the last row is not converted
    Intrinsic intrinsic_; // intrinsic parameters
    std::vector<Extrinsic, Eigen::aligned_allocator<Extrinsic>> extrinsics_; // extrinsic parameters

    // for jetson design and usage
    float* unified_float_intrinsic_; // allocated in unified memory
    float* unified_float_extrinsic_; // allocated in unified memory, the same organization as float_extrinsic_ member, but in jetson, this is the only thing that is activated for extrinsic
    // directly in world to camera format
    int number_extrinsics;
};

}; // namespace Camera

#endif // CAM_H