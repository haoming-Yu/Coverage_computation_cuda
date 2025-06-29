#ifndef CUDA_COVERAGE_H
#define CUDA_COVERAGE_H

#include <cuda_runtime.h>

namespace Coverage {

/**
 * @brief device function: check if a 3D point is visible in the camera's field of view.
 *
 * The function performs the following steps:
 * 1. Convert the 3D point from the world coordinate system to the camera coordinate system.
 * 2. Project the 3D point in the camera coordinate system to the 2D image plane.
 * 3. Check if the projected 2D coordinates are within the image boundaries.
 * 4. Check if the point is in front of the camera (Z > 0).
 *
 * @param point_world   3D point in the world coordinate system.
 * @param d_R_cw       device version of R_cw (world to camera)
 * @param d_t_cw       device version of t_cw (world to camera)
 * @param fx           focal length in x direction
 * @param fy           focal length in y direction
 * @param cx           principal point in x direction
 * @param cy           principal point in y direction
 * @param img_width    image width
 * @param img_height   image height
 * @return true if the point is visible, false otherwise.
 */
__device__ bool is_point_visible(
    const float3& point_world,
    const float* d_R_cw,
    const float* d_t_cw,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float img_width,
    const float img_height);

/**
 * @brief CUDA Kernel function: generate visibility matrix.
 *
 * Use 2D thread grid to parallel process cameras and points.
 * Each thread calculates the visibility of a specific point for a specific camera, and stores the result in the visibility matrix.
 *
 * @param intrinsics                camera intrinsics
 * @param d_extrinsics_array        camera extrinsics
 * @param d_points_array            3D points
 * @param d_visibility_matrix       output visibility matrix
 * @param num_cameras               number of cameras
 * @param num_points                number of points
 */
__global__ void generateVisibilityMatrixKernel(
    const float* __restrict__ intrinsics, // camera intrinsics
    const float* __restrict__ d_extrinsics_array, // camera extrinsics
    const float3* __restrict__ d_points_array, // 3D points
    unsigned char* __restrict__ d_visibility_matrix, // output visibility matrix
    int num_cameras, // number of cameras
    int num_points); // number of points


/**
 * @brief CUDA Kernel function: find candidate cameras for final coverage selection.
 *
 * Get the final candidate cameras for later texture usage.
 *
 * @param d_visibility_matrix            visibility matrix, each row is a camera, each column is a point, and flattened to a 1D array
 * @param d_camera_visible_points_count  camera visible points count, each element is the number of visible points for a camera
 * @param d_point_candidate_camera_index point candidate camera index, each element is the index of the candidate camera for a point (greedily select the camera with the most visible points to maximize each camera usage)
 * @param d_candidate_camera_mask        candidate camera mask, each element is 1 if the camera is a candidate, 0 otherwise, the size is num_cameras. Need to be initialized to 0 before calling the kernel.
 * @param num_cameras                    number of cameras
 * @param num_points                     number of points
 */
__global__ void findCandidateCameraKernel(
    const unsigned char* __restrict__ d_visibility_matrix,
    const int* __restrict__ d_camera_visible_points_count,
    int* __restrict__ d_point_candidate_camera_index,
    unsigned int* __restrict__ d_candidate_camera_mask, // This pointer points to pinned memory accessible by CPU
    int num_cameras,
    int num_points);

/**
 * @brief CPU Interface: generate visibility matrix.
 *
 * A CPU wrapper for CUDA Kernel
 * 
 * @param intrinsic                 camera intrinsics
 * @param extrinsics_array        camera extrinsics
 * @param d_points_array            3D points
 * @param d_visibility_matrix       output visibility matrix
 * @param num_cameras               number of cameras
 * @param num_points                number of points
 */
void getVisibilityMatrixKernel(
    const float* intrinsic,   // camera intrinsic, in the format of a float*, (0, 1, 2, 3, 4, 5) -> (fx, fy, cx, cy, img_width, img_height)
    const float* extrinsics,  // camera extrinsics, in the format of a float*, 12 elements a group, records the element of T matrix from (0, 0) to (2, 3)
    const float3* points,     // point coordinates, in the format of float3*
    int num_cameras,          // camera number, also group number of extrinsics array
    int num_points,           // point number, also length of points array
    unsigned char* visibility_matrix,   // now leave it for debugging, for better design, it shouldn't be here, just pass the data on GPU, do not back load the data to CPU
    unsigned int* candidate_camera_mask // candidate camera mask, each element is 1 if the camera is a candidate, 0 otherwise, the size is num_cameras.
);

} // namespace Coverage

#endif // CUDA_COVERAGE_H