#include "mesh.h"
#include "camera.h"
#include "depth.h"
#include "cudaCoverage.cuh"
#include <cuda_runtime.h>
#include <iostream> 
#include <fstream>

// Simple helper to print usage information
void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " -m <mesh_file> -i <intrinsic_file> -e <extrinsic_file> -d <depth_directory>\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -m <mesh_file>        Path to the mesh .ply file\n";
    std::cout << "  -i <intrinsic_file>   Path to the camera intrinsic log file\n";
    std::cout << "  -e <extrinsic_file>   Path to the camera extrinsic trajectory log file\n";
    std::cout << "  -d <depth_directory>  Directory containing depth maps\n";
    std::cout << "  -h, --help            Show this help message and exit\n";
}

int main(int argc, char** argv) {
    // Default empty paths – will be filled from CLI
    std::string mesh_path;
    std::string intrinsic_path;
    std::string extrinsic_path;
    std::string depth_dir;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-m" && i + 1 < argc) {
            mesh_path = argv[++i];
        } else if (arg == "-i" && i + 1 < argc) {
            intrinsic_path = argv[++i];
        } else if (arg == "-e" && i + 1 < argc) {
            extrinsic_path = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {
            depth_dir = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    // Validate that all required paths are provided
    if (mesh_path.empty() || intrinsic_path.empty() || extrinsic_path.empty() || depth_dir.empty()) {
        std::cerr << "Error: Missing required arguments." << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    MeshProcessing::Mesh mesh;
    mesh.loadFromFile(mesh_path);
    std::vector<float3> vertexes = mesh.getVertexes();
    std::cout << "Vertexes number rechecked: " << vertexes.size() << std::endl;

    Camera::Cam cam;
    cam.loadIntrinsic(intrinsic_path);
    cam.loadExtrinsics(extrinsic_path);
    cam.dump_intrinsic_to_float(); // prepare for the data loading of gpu
    // dump has been checked, and the data is correct
    cam.dump_extrinsic_to_float();

    Depth::Depth depth;
    depth.loadDepthMaps(depth_dir);
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
    Coverage::getVisibilityMatrixKernel(cam.float_intrinsic_, cam.float_extrinsic_, vertexes.data(), depth.depth_maps_float_, num_cameras, num_points, visibility_matrix, candidate_camera_mask);
    std::cout << "coverage selection finished" << std::endl;

    // for debugging, to check the GPU performance
    // for (int test = 0; test < 5000; test++) {
    //     Coverage::getVisibilityMatrixKernel(cam.float_intrinsic_, cam.float_extrinsic_, vertexes.data(), depth.depth_maps_float_, num_cameras, num_points, visibility_matrix, candidate_camera_mask);
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