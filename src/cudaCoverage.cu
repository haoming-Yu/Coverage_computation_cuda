#include "cudaCoverage.cuh"
#include "cudaUtils.cuh"
#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>      // For thrust::reduce
#include <thrust/functional.h>  // For thrust::plus

void Coverage::getVisibilityMatrixKernel(
    const float* intrinsic,   // camera intrinsic, in the format of a float*, (0, 1, 2, 3, 4, 5) -> (fx, fy, cx, cy, img_width, img_height)
    const float* extrinsics,  // camera extrinsics, in the format of a float*, 12 elements a group, records the element of T matrix from (0, 0) to (2, 3)
    const float3* points,     // point coordinates, in the format of float3*
    const float* depth_maps,  // depth maps' array, the sequence is: depth_map number, rows, cols
    int num_cameras,          // camera number, also group number of extrinsics array
    int num_points,           // point number, also length of points array
    unsigned char* visibility_matrix, // now leave it for debugging, for better design, it shouldn't be here, just pass the data on GPU, do not back load the data to CPU
    unsigned int* candidate_camera_mask // candidate camera mask, each element is 1 if the camera is a candidate, 0 otherwise, the size is num_cameras.
    
) {
    // 1. generate visibility matrix
    // first, allocate memory on GPU for visibility matrix
    unsigned char* visibility_matrix_gpu;
    CUDA_CHECK(cudaMalloc((void**)&visibility_matrix_gpu, num_cameras * num_points * sizeof(unsigned char)));
    
    // then, allocate memory on GPU for elements needed to be uploaded
    float* intrinsic_gpu;
    CUDA_CHECK(cudaMalloc((void**)&intrinsic_gpu, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(intrinsic_gpu, intrinsic, 6 * sizeof(float), cudaMemcpyHostToDevice));

    float* extrinsics_gpu;
    CUDA_CHECK(cudaMalloc((void**)&extrinsics_gpu, num_cameras * 12 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(extrinsics_gpu, extrinsics, num_cameras * 12 * sizeof(float), cudaMemcpyHostToDevice));

    float3* points_gpu;
    CUDA_CHECK(cudaMalloc((void**)&points_gpu, num_points * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(points_gpu, points, num_points * sizeof(float3), cudaMemcpyHostToDevice));

    float* depth_maps_gpu;
    // rows and cols are the width and height of the depth map, which is the same as the image size
    // rows can be obtained from image_width, intrinsic[4]
    // cols can be obtained from image_height, intrinsic[5]
    CUDA_CHECK(cudaMalloc((void**)&depth_maps_gpu, num_cameras * intrinsic[4] * intrinsic[5] * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(depth_maps_gpu, depth_maps, num_cameras * intrinsic[4] * intrinsic[5] * sizeof(float), cudaMemcpyHostToDevice));

    // for debugging, print the depth maps' size
    std::cout << "depth maps' size: " << intrinsic[4] << "x" << intrinsic[5] << std::endl;
    
    size_t total_combinations = (size_t)num_cameras * num_points;
    // prepare to call the kernel function
    int threadsPerBlock = 1024;
    int gridSize = (total_combinations + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Start to launch CUDA Kernel\n");
    printf("  grid size: %d\n", gridSize);
    printf("  threads per block: %d\n", threadsPerBlock);
    printf("  total threads: %llu\n", (long long)gridSize * threadsPerBlock);

    // record elapsed time for generating visibility matrix
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    // call the kernel function to generate visibility matrix
    Coverage::generateVisibilityMatrixKernel<<<gridSize, threadsPerBlock>>>(intrinsic_gpu, extrinsics_gpu, points_gpu, depth_maps_gpu, visibility_matrix_gpu, num_cameras, num_points);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Kernel execution time for generating visibility matrix: %f ms\n", elapsedTime);

    // 2. count the number of visible points for each camera using thrust
    thrust::device_vector<int> d_camera_visible_points_count_thrust(num_cameras);
    std::cout << "\nStarting Thrust to calculate camera visible points count (using existing GPU memory)..." << std::endl;

    cudaEvent_t start_thrust_count, stop_thrust_count;
    CUDA_CHECK(cudaEventCreate(&start_thrust_count));
    CUDA_CHECK(cudaEventCreate(&stop_thrust_count));

    CUDA_CHECK(cudaEventRecord(start_thrust_count));
    // Loop through each camera (row) and use thrust::reduce to sum the visible points
    for (int camera_index = 0; camera_index < num_cameras; camera_index++) {
        thrust::device_ptr<const unsigned char> row_start = thrust::device_pointer_cast(visibility_matrix_gpu + camera_index * num_points);
        thrust::device_ptr<const unsigned char> row_end = row_start + num_points;

        int sum_for_camera = thrust::reduce(row_start, row_end, 0, thrust::plus<int>());
        d_camera_visible_points_count_thrust[camera_index] = sum_for_camera;   
    }
    CUDA_CHECK(cudaEventRecord(stop_thrust_count));
    CUDA_CHECK(cudaEventSynchronize(stop_thrust_count));
    
    float thrust_count_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&thrust_count_milliseconds, start_thrust_count, stop_thrust_count));
    std::cout << "Thrust calculateCameraVisiblePointsCount execution time: " << thrust_count_milliseconds << " ms" << std::endl;

    // get the raw pointer of camera visible points count array on GPU, the size is num_cameras
    int* d_camera_visible_points_count_ptr = thrust::raw_pointer_cast(d_camera_visible_points_count_thrust.data());

    // 3. find candidate cameras for final coverage selection
    cudaEvent_t start_candidate, stop_candidate;
    CUDA_CHECK(cudaEventCreate(&start_candidate));
    CUDA_CHECK(cudaEventCreate(&stop_candidate));

    int* d_point_candidate_camera_index_gpu;
    CUDA_CHECK(cudaMalloc((void**)&d_point_candidate_camera_index_gpu, num_points * sizeof(int)));
    unsigned int* d_candidate_camera_mask_gpu;
    CUDA_CHECK(cudaMalloc((void**)&d_candidate_camera_mask_gpu, num_cameras * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_candidate_camera_mask_gpu, 0, num_cameras * sizeof(unsigned int)));

    int candidate_blockSize = 256;
    int candidate_gridSize = (num_points + candidate_blockSize - 1) / candidate_blockSize;

    std::cout << "\nStarting CUDA Kernel to find candidate cameras..." << std::endl;
    std::cout << "    Grid size: " << candidate_gridSize << ", block size: " << candidate_blockSize << std::endl;

    CUDA_CHECK(cudaEventRecord(start_candidate));
    findCandidateCameraKernel<<<candidate_gridSize, candidate_blockSize>>>(
        visibility_matrix_gpu,
        d_camera_visible_points_count_ptr,
        d_point_candidate_camera_index_gpu,
        d_candidate_camera_mask_gpu,
        num_cameras,
        num_points);
    CUDA_CHECK(cudaEventRecord(stop_candidate));
    CUDA_CHECK(cudaEventSynchronize(stop_candidate));

    float candidate_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&candidate_milliseconds, start_candidate, stop_candidate));
    std::cout << "Kernel execution time for finding candidate cameras: " << candidate_milliseconds << " ms" << std::endl;

    // copy d_candidate_camera_mask_gpu to CPU
    CUDA_CHECK(cudaMemcpy(candidate_camera_mask, d_candidate_camera_mask_gpu, num_cameras * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::cout << "Downloaded candidate camera mask to CPU" << std::endl;

    // finally, copy the data back to CPU
    CUDA_CHECK(cudaMemcpy(visibility_matrix, visibility_matrix_gpu, num_cameras * num_points * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // finally, free the memory on GPU
    CUDA_CHECK(cudaFree(visibility_matrix_gpu));
    CUDA_CHECK(cudaFree(intrinsic_gpu));
    CUDA_CHECK(cudaFree(extrinsics_gpu));
    CUDA_CHECK(cudaFree(points_gpu));
    CUDA_CHECK(cudaFree(d_point_candidate_camera_index_gpu));
    CUDA_CHECK(cudaFree(d_candidate_camera_mask_gpu));
    CUDA_CHECK(cudaFree(depth_maps_gpu));
}

__device__ bool Coverage::is_point_visible(
    const float3& point_world,
    const int camera_idx,
    const float* d_R_cw,
    const float* d_t_cw,

    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float img_width,
    const float img_height,
    const float* d_depth_maps) {
    // 1. convert the 3D point from the world coordinate system to the camera coordinate system
    // manually calculate the matrix multiplication
    float3 point_camera;
    point_camera.x = d_R_cw[0] * point_world.x + d_R_cw[1] * point_world.y + d_R_cw[2] * point_world.z + d_t_cw[0];
    point_camera.y = d_R_cw[3] * point_world.x + d_R_cw[4] * point_world.y + d_R_cw[5] * point_world.z + d_t_cw[1];
    point_camera.z = d_R_cw[6] * point_world.x + d_R_cw[7] * point_world.y + d_R_cw[8] * point_world.z + d_t_cw[2];
    
    // 2. check if the point is in front of the camera
    if (point_camera.z <= 0) {
        return false;
    }
    
    // 3. project 3D point to uv coordinates, u = fx * (px_cam / pz_cam) + cx, v = fy * (py_cam / pz_cam) + cy
    float u = fx * (point_camera.x / point_camera.z) + cx;
    float v = fy * (point_camera.y / point_camera.z) + cy;

    int u_int = (int)u;
    int v_int = (int)v;

    // test if the point is within the image boundaries
    if (u_int < 0 || u_int >= img_width || v_int < 0 || v_int >= img_height) {
        return false;
    }

    // 4. get current depth value
    int index_approximate_depth_map = camera_idx * img_width * img_height + v_int * img_width + u_int;
    float current_depth = d_depth_maps[index_approximate_depth_map];

    if (fabs(current_depth - point_camera.z) > DEPTH_MAP_THRESHOLD) {
        return false;
    }
    
    // 5. check if the point is within the image boundaries
    return (u >= 0 && u < img_width && v >= 0 && v < img_height);
}   

__global__ void Coverage::generateVisibilityMatrixKernel(
    const float* __restrict__ d_intrinsic, // camera intrinsic, fx, fy, cx, cy, img_width, img_height
    const float* __restrict__ d_extrinsics_array, // camera extrinsics
    const float3* __restrict__ d_points_array, // 3D points
    const float* __restrict__ d_depth_maps, // depth maps array
    unsigned char* __restrict__ d_visibility_matrix, // output visibility matrix
    int num_cameras, // number of cameras
    int num_points) { // number of points

    // 1. get camera index and point index
    size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_combinations = (size_t)num_cameras * num_points;

    for (; global_index < total_combinations; global_index += blockDim.x * gridDim.x) {
        int camera_index = global_index / num_points;
        int point_index = global_index % num_points;

        // for debugging, print the critical informations, ensure the information is printed once
        // if (camera_index == 0 && point_index == 0) {
        //     printf("intrinsic on gpu: %f, %f, %f, %f, %f, %f, should be interpreted as fx, fy, cx, cy, img_width, img_height\n", d_intrinsic[0], d_intrinsic[1], d_intrinsic[2], d_intrinsic[3], d_intrinsic[4], d_intrinsic[5]);
        //     printf("first two extrinsics on gpu: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", d_extrinsics_array[0], d_extrinsics_array[1], d_extrinsics_array[2], d_extrinsics_array[3], d_extrinsics_array[4], d_extrinsics_array[5], d_extrinsics_array[6], d_extrinsics_array[7], d_extrinsics_array[8], d_extrinsics_array[9], d_extrinsics_array[10], d_extrinsics_array[11]);
        //     printf("   second extrinsics on gpu: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", d_extrinsics_array[12], d_extrinsics_array[13], d_extrinsics_array[14], d_extrinsics_array[15], d_extrinsics_array[16], d_extrinsics_array[17], d_extrinsics_array[18], d_extrinsics_array[19], d_extrinsics_array[20], d_extrinsics_array[21], d_extrinsics_array[22], d_extrinsics_array[23]);
        //     printf("first two points(world coordinates) on gpu: %f, %f, %f, %f, %f, %f\n", d_points_array[0].x, d_points_array[0].y, d_points_array[0].z, d_points_array[1].x, d_points_array[1].y, d_points_array[1].z);
        // }
        // 2. get camera extrinsics, get the rotation matrix and translation vector manually
        // for debugging, print the critical informations, ensure the information is printed once
        // if (camera_index == 1 && point_index == 0) {
        //     printf("R_cw on gpu: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", d_R_cw[0], d_R_cw[1], d_R_cw[2], d_R_cw[3], d_R_cw[4], d_R_cw[5], d_R_cw[6], d_R_cw[7], d_R_cw[8]);
        //     printf("t_cw on gpu: %f, %f, %f\n", d_t_cw[0], d_t_cw[1], d_t_cw[2]);
        // }
        // 3. get 3D point
        // if (camera_index == 0 && point_index == 0) {
        //     printf("First point's coordinates: (%f, %f, %f)\n", point_world.x, point_world.y, point_world.z);
        // }
        // float3 point_camera;
        // point_camera.x = d_R_cw[0] * point_world.x + d_R_cw[1] * point_world.y + d_R_cw[2] * point_world.z + d_t_cw[0];
        // point_camera.y = d_R_cw[3] * point_world.x + d_R_cw[4] * point_world.y + d_R_cw[5] * point_world.z + d_t_cw[1];
        // point_camera.z = d_R_cw[6] * point_world.x + d_R_cw[7] * point_world.y + d_R_cw[8] * point_world.z + d_t_cw[2];
        // if (1) {
        //     // 3. project 3D point to uv coordinates, u = fx * (px_cam / pz_cam) + cx, v = fy * (py_cam / pz_cam) + cy
        //     float u = d_intrinsic[0] * (point_camera.x / point_camera.z) + d_intrinsic[2];
        //     float v = d_intrinsic[1] * (point_camera.y / point_camera.z) + d_intrinsic[3];
        //     printf("u: %f, v: %f, z: %f\n", u, v, point_camera.z);
        // }
        // for debugging, prechecking z value in camera coordinate system
        // float3 point_camera;
        // point_camera.x = d_R_cw[0] * point_world.x + d_R_cw[1] * point_world.y + d_R_cw[2] * point_world.z + d_t_cw[0];
        // point_camera.y = d_R_cw[3] * point_world.x + d_R_cw[4] * point_world.y + d_R_cw[5] * point_world.z + d_t_cw[1];
        // point_camera.z = d_R_cw[6] * point_world.x + d_R_cw[7] * point_world.y + d_R_cw[8] * point_world.z + d_t_cw[2];
        // if (point_camera.z > 0) {
        //     printf("point %d is visible in camera %d\n", point_index, camera_index);
        // }
        // if (camera_index == 10 && point_index == 94775) {
        //     printf("camera 10's rotation: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", d_R_cw[0], d_R_cw[1], d_R_cw[2], d_R_cw[3], d_R_cw[4], d_R_cw[5], d_R_cw[6], d_R_cw[7], d_R_cw[8]);
        //     printf("camera 10's translation: %f, %f, %f\n", d_t_cw[0], d_t_cw[1], d_t_cw[2]);
        //     printf("point world coordinates: %f, %f, %f\n", point_world.x, point_world.y, point_world.z);
        //     printf("point camera coordinates: %f, %f, %f\n", point_camera.x, point_camera.y, point_camera.z);
        // }
        // 4. check if the point is visible
        
        const float* d_T_cw = d_extrinsics_array + camera_index * 12;
        float d_R_cw[9];
        float d_t_cw[3];
        
        d_R_cw[0] = d_T_cw[0];
        d_R_cw[1] = d_T_cw[1];
        d_R_cw[2] = d_T_cw[2];
        d_R_cw[3] = d_T_cw[4];
        d_R_cw[4] = d_T_cw[5];
        d_R_cw[5] = d_T_cw[6];
        d_R_cw[6] = d_T_cw[8];
        d_R_cw[7] = d_T_cw[9];
        d_R_cw[8] = d_T_cw[10];

        d_t_cw[0] = d_T_cw[3];
        d_t_cw[1] = d_T_cw[7];
        d_t_cw[2] = d_T_cw[11];

        const float3& point_world = d_points_array[point_index];

        bool is_visible = is_point_visible(point_world, camera_index, d_R_cw, d_t_cw, d_intrinsic[0], d_intrinsic[1], d_intrinsic[2], d_intrinsic[3], d_intrinsic[4], d_intrinsic[5], d_depth_maps);
        // for debugging, get visible points
        // if (is_visible) {
        //     printf("point %d is visible in camera %d\n", point_index, camera_index);
        // }
        d_visibility_matrix[camera_index * num_points + point_index] = (is_visible ? 1 : 0);
    }
}

__global__ void Coverage::findCandidateCameraKernel(
    const unsigned char* __restrict__ d_visibility_matrix,
    const int* __restrict__ d_camera_visible_points_count,
    int* __restrict__ d_point_candidate_camera_index,
    unsigned int* __restrict__ d_candidate_camera_mask, // This pointer points to pinned memory accessible by CPU
    int num_cameras,
    int num_points) {
    // 1. get point index
    size_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_points = (size_t)num_points;

    // incase the point index is bigger than the number of allocable threads, use the following loop to ensure the point is processed
    for (; point_index < total_points; point_index += blockDim.x * gridDim.x) {
        int max_visible_count_for_point = -1; // stores the maximum numbr of visible points among cameras visible to the current point, -1 as initial value
        int candidate_camera_idx_for_pt = -1; // stores the index of the candidate camera chosen for the current point, -1 means no camera is chosen for the point

        for (int camera_idx = 0; camera_idx < num_cameras; camera_idx++) {
            if (d_visibility_matrix[camera_idx * num_points + point_index] == 1) {
                // the camera is visible to the point, check the camera's visible points count
                int current_camera_visible_count = d_camera_visible_points_count[camera_idx];

                if (current_camera_visible_count > max_visible_count_for_point ||
                    (current_camera_visible_count == max_visible_count_for_point && 
                    camera_idx < candidate_camera_idx_for_pt)) {
                    max_visible_count_for_point = current_camera_visible_count;
                    candidate_camera_idx_for_pt = camera_idx;
                }
            }
        }

        // store the index of the candidate camera chosen for the current point in the output array
        // actually not necessary, just for debugging
        d_point_candidate_camera_index[point_index] = candidate_camera_idx_for_pt;

        // update the candidate camera mask
        if (candidate_camera_idx_for_pt != -1) {
            // ensure correctness
            atomicOr(&d_candidate_camera_mask[candidate_camera_idx_for_pt], static_cast<unsigned int>(1));
            
            // another option, direct assignment, not tested.
            // d_candidate_camera_mask[candidate_camera_idx_for_pt] = 1;
        }
    }
}