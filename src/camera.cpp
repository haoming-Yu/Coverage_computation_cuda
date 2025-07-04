#include "camera.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iostream>

namespace Camera {

Cam::Cam() {}

void Cam::loadIntrinsic(const std::string& intrinsic_file) {
    std::ifstream file(intrinsic_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open intrinsic file: " + intrinsic_file);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        float value;
        if (iss >> key >> value) {
            if (key == "fx") {
                intrinsic_.fx = value;
            } else if (key == "fy") {
                intrinsic_.fy = value;
            } else if (key == "cx") {
                intrinsic_.cx = value;
            } else if (key == "cy") {
                intrinsic_.cy = value;
            } else if (key == "width") {
                intrinsic_.img_width = static_cast<int>(value);
            } else if (key == "height") {
                intrinsic_.img_height = static_cast<int>(value);
            }
        }
    }
    file.close();
}

void Cam::loadIntrinsic_jetson(const std::string& intrinsic_file) {
    std::ifstream file(intrinsic_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open intrinsic file: " + intrinsic_file);
    }

    std::string line;
    CUDA_CHECK(cudaMallocManaged(&this->unified_float_intrinsic_, sizeof(float) * 6));
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        float value;
        if (iss >> key >> value) {
            if (key == "fx") {
                this->unified_float_intrinsic_[0] = value;
            } else if (key == "fy") {
                this->unified_float_intrinsic_[1] = value;
            } else if (key == "cx") {
                this->unified_float_intrinsic_[2] = value;
            } else if (key == "cy") {
                this->unified_float_intrinsic_[3] = value;
            } else if (key == "width") {
                this->unified_float_intrinsic_[4] = value;
            } else if (key == "height") {
                this->unified_float_intrinsic_[5] = value;
            }
        }
    }
    file.close();
}

void Cam::loadExtrinsics_colormap(const std::string& extrinsic_file) {
    std::ifstream file(extrinsic_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open extrinsic file: " + extrinsic_file);
    }

    std::string line;
    int line_num = 0;
    float t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        switch (line_num % 5) {
            case 0:
                break;
            case 1:
                iss >> t00 >> t01 >> t02 >> t03;
                break;
            case 2:
                iss >> t10 >> t11 >> t12 >> t13;
                break;
            case 3:
                iss >> t20 >> t21 >> t22 >> t23;
                break;
            case 4:
                iss >> t30 >> t31 >> t32 >> t33;
                Extrinsic extrinsic;
                extrinsic.T_wc << t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33;
                extrinsic.R_wc = extrinsic.T_wc.block<3, 3>(0, 0);
                extrinsic.t_wc = extrinsic.T_wc.block<3, 1>(0, 3);
                extrinsic.T_cw = extrinsic.T_wc.inverse();
                extrinsic.R_cw = extrinsic.T_cw.block<3, 3>(0, 0);
                extrinsic.t_cw = extrinsic.T_cw.block<3, 1>(0, 3);

                extrinsics_.push_back(extrinsic);
                /* for debug */
                /* std::cout << "Extrinsic " << line_num / 5 << ":\n" << extrinsic.T_wc << std::endl; */
                break;
        }
        line_num++;
    }

    file.close();
}

void Cam::loadExtrinsics_colormap_jetson(const std::string& extrinsic_file) {
    std::ifstream file(extrinsic_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open extrinsic file: " + extrinsic_file);
    }

    std::string line;
    std::vector<Extrinsic_jetson> extrinsics_jetson;
    int line_num = 0;
    float t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        switch (line_num % 5) {
            case 0:
                break;
            case 1:
                iss >> t00 >> t01 >> t02 >> t03;
                break;
            case 2:
                iss >> t10 >> t11 >> t12 >> t13;
                break;
            case 3:
                iss >> t20 >> t21 >> t22 >> t23;
                break;
            case 4:
                iss >> t30 >> t31 >> t32 >> t33;
                
                Eigen::Matrix4f T_wc;
                T_wc << t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33;
                Eigen::Matrix4f T_cw = T_wc.inverse();
                Extrinsic_jetson extrinsic_jetson;
                extrinsic_jetson.t00 = T_cw(0, 0);
                extrinsic_jetson.t01 = T_cw(0, 1);
                extrinsic_jetson.t02 = T_cw(0, 2);
                extrinsic_jetson.t03 = T_cw(0, 3);
                extrinsic_jetson.t10 = T_cw(1, 0);
                extrinsic_jetson.t11 = T_cw(1, 1);
                extrinsic_jetson.t12 = T_cw(1, 2);
                extrinsic_jetson.t13 = T_cw(1, 3);
                extrinsic_jetson.t20 = T_cw(2, 0);
                extrinsic_jetson.t21 = T_cw(2, 1);
                extrinsic_jetson.t22 = T_cw(2, 2);
                extrinsic_jetson.t23 = T_cw(2, 3);

                extrinsics_jetson.push_back(extrinsic_jetson);
                /* for debug */
                /* std::cout << "Extrinsic " << line_num / 5 << ":\n" << extrinsic.T_wc << std::endl; */
                break;
        }
        line_num++;
    }

    file.close();

    // allocate unified memory for extrinsics_jetson
    this->number_extrinsics = extrinsics_jetson.size(); // number of extrinsics
    CUDA_CHECK(cudaMallocManaged(&this->unified_float_extrinsic_, sizeof(float) * 12 * this->number_extrinsics));
    for (int i = 0; i < this->number_extrinsics; i++) {
        this->unified_float_extrinsic_[i * 12 + 0] = extrinsics_jetson[i].t00;
        this->unified_float_extrinsic_[i * 12 + 1] = extrinsics_jetson[i].t01;
        this->unified_float_extrinsic_[i * 12 + 2] = extrinsics_jetson[i].t02;
        this->unified_float_extrinsic_[i * 12 + 3] = extrinsics_jetson[i].t03;
        this->unified_float_extrinsic_[i * 12 + 4] = extrinsics_jetson[i].t10;
        this->unified_float_extrinsic_[i * 12 + 5] = extrinsics_jetson[i].t11;
        this->unified_float_extrinsic_[i * 12 + 6] = extrinsics_jetson[i].t12;
        this->unified_float_extrinsic_[i * 12 + 7] = extrinsics_jetson[i].t13;
        this->unified_float_extrinsic_[i * 12 + 8] = extrinsics_jetson[i].t20;
        this->unified_float_extrinsic_[i * 12 + 9] = extrinsics_jetson[i].t21;
        this->unified_float_extrinsic_[i * 12 + 10] = extrinsics_jetson[i].t22;
        this->unified_float_extrinsic_[i * 12 + 11] = extrinsics_jetson[i].t23;
    }

    // free the extrinsics_jetson as soon as possible
    extrinsics_jetson.clear();
    extrinsics_jetson.shrink_to_fit();
}

};