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

};