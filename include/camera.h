#ifndef CAM_H
#define CAM_H

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

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

class Cam {
public:
    Cam();
    ~Cam() {};
    void loadIntrinsic(const std::string& intrinsic_file);
    void loadExtrinsics(const std::string& extrinsic_file);
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
public:
    float* float_intrinsic_;
    float* float_extrinsic_; // 3 * 4 world to camera extrinsics, to save space, the last row is not converted
    Intrinsic intrinsic_; // intrinsic parameters
    std::vector<Extrinsic, Eigen::aligned_allocator<Extrinsic>> extrinsics_; // extrinsic parameters
};

}; // namespace Camera

#endif // CAM_H