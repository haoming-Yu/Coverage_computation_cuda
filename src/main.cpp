#include "mesh.h"
#include "camera.h"
#include "depth.h"
#include "cudaCoverage.cuh"
#include <cuda_runtime.h>
#include <iostream> 
#include <fstream>

int main() {
    MeshProcessing::Mesh mesh;
    mesh.loadFromFile("../data/underground/mesh/filtered_mesh.ply");
    std::vector<float3> vertexes = mesh.getVertexes();
    std::cout << "Vertexes number rechecked: " << vertexes.size() << std::endl;

    Camera::Cam cam;
    cam.loadIntrinsic("../data/underground/intrinsic.log");
    cam.loadExtrinsics("../data/underground/traj.log");
    cam.dump_intrinsic_to_float(); // prepare for the data loading of gpu
    // dump has been checked, and the data is correct
    cam.dump_extrinsic_to_float();

    Depth::Depth depth;
    depth.loadDepthMaps("../data/underground/depth");
    depth.convertToFloat();

    // for debugging, to check the camera extrinsics number 10
    // std::cout << "camera 10's extrinsic: " << std::endl;
    // std::cout << cam.extrinsics_[10].T_cw << std::endl;

    // for convenience
    int num_cameras = cam.extrinsics_.size();
    int num_points = vertexes.size();

    unsigned char* visibility_matrix;
    visibility_matrix = new unsigned char[num_cameras * num_points];

    unsigned int* candidate_camera_mask = new unsigned int[num_cameras];
    Coverage::getVisibilityMatrixKernel(cam.float_intrinsic_, cam.float_extrinsic_, vertexes.data(), num_cameras, num_points, visibility_matrix, candidate_camera_mask);
    std::cout << "coverage selection finished" << std::endl;

    // for debugging, to check the GPU performance
    // for (int test = 0; test < 5000; test++) {
    //     Coverage::getVisibilityMatrixKernel(cam.float_intrinsic_, cam.float_extrinsic_, vertexes.data(), num_cameras, num_points, visibility_matrix, candidate_camera_mask);
    // }

    // for debugging, check how many cameras are selected as candidate
    int num_candidate_cameras = 0;
    for (int i = 0; i < num_cameras; i++) {
        if (candidate_camera_mask[i] == 1) {
            num_candidate_cameras++;
        }
    }
    std::cout << "number of candidate cameras: " << num_candidate_cameras << std::endl;

    // for debugging, check the correctness of the visibility matrix
    // int none_zero_elements = 0;
    // for (int i = 0; i < num_points * num_cameras; i++) {
    //     if (visibility_matrix[i] == 1) {
    //         none_zero_elements++;
    //     }
    // }
    // std::cout << "none_zero_elements: " << none_zero_elements << std::endl;
    // std::ofstream outfile("../data/desk/visibility_matrix.txt");
    // if (outfile.is_open()) {
    //     for (int i = 0; i < num_cameras; ++i) {
    //         for (int j = 0; j < num_points; ++j) {
    //             outfile << static_cast<int>(visibility_matrix[i * num_points + j]) << " ";
    //         }
    //         outfile << "\n";
    //     }
    //     outfile.close();
    //     std::cout << "Visibility matrix written to visibility_matrix.txt" << std::endl;
    // } else {
    //     std::cerr << "Unable to open file for writing" << std::endl;
    // }

    return 0;
}   