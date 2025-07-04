#include "cudaCoverage.cuh"
#include "cudaUtils.cuh"

#include <cublas_v2.h>

#include <stdio.h>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/reduce.h>      // For thrust::reduce
// #include <thrust/functional.h>  // For thrust::plus

void Coverage::getVisibilityMatrixKernel(
    const float* intrinsic,   // camera intrinsic, in the format of a float*, (0, 1, 2, 3, 4, 5) -> (fx, fy, cx, cy, img_width, img_height)
    const float* extrinsics,  // camera extrinsics, in the format of a float*, 12 elements a group, records the element of T matrix from (0, 0) to (2, 3)
    const float3* points,     // point coordinates, in the format of float3*
    const float* depth_maps,  // depth maps' array, the sequence is: depth_map number, rows, cols
    int num_cameras,          // camera number, also group number of extrinsics array
    int num_points,           // point number, also length of points array
    float* visibility_matrix, // now leave it for debugging, for better design, it shouldn't be here, just pass the data on GPU, do not back load the data to CPU
    unsigned int* candidate_camera_mask, // candidate camera mask, each element is 1 if the camera is a candidate, 0 otherwise, the size is num_cameras.
    int* point_candidate_camera_index // point candidate camera index, each element is the index of the candidate camera for the point, -1 means no camera is chosen for the point, the size is num_points.
) {
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
    Coverage::generateVisibilityMatrixKernel<<<gridSize, threadsPerBlock>>>(intrinsic, extrinsics, points, depth_maps, visibility_matrix, num_cameras, num_points);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Kernel execution time for generating visibility matrix: %f ms\n", elapsedTime);

    // // 2. count the number of visible points for each camera using thrust
    // thrust::device_vector<int> d_camera_visible_points_count_thrust(num_cameras);
    // std::cout << "\nStarting Thrust to calculate camera visible points count (using existing GPU memory)..." << std::endl;

    // cudaEvent_t start_thrust_count, stop_thrust_count;
    // CUDA_CHECK(cudaEventCreate(&start_thrust_count));
    // CUDA_CHECK(cudaEventCreate(&stop_thrust_count));

    // CUDA_CHECK(cudaEventRecord(start_thrust_count));
    // // Loop through each camera (row) and use thrust::reduce to sum the visible points
    // for (int camera_index = 0; camera_index < num_cameras; camera_index++) {
    //     thrust::device_ptr<const float> row_start = thrust::device_pointer_cast(visibility_matrix + camera_index * num_points);
    //     thrust::device_ptr<const float> row_end = row_start + num_points;

    //     int sum_for_camera = thrust::reduce(row_start, row_end, 0, thrust::plus<int>());
    //     d_camera_visible_points_count_thrust[camera_index] = sum_for_camera;   
    // }
    // CUDA_CHECK(cudaEventRecord(stop_thrust_count));
    // CUDA_CHECK(cudaEventSynchronize(stop_thrust_count));
    
    // float thrust_count_milliseconds = 0;
    // CUDA_CHECK(cudaEventElapsedTime(&thrust_count_milliseconds, start_thrust_count, stop_thrust_count));
    // std::cout << "Thrust calculateCameraVisiblePointsCount execution time: " << thrust_count_milliseconds << " ms" << std::endl;

    // // get the raw pointer of camera visible points count array on GPU, the size is num_cameras
    // int* d_camera_visible_points_count_ptr = thrust::raw_pointer_cast(d_camera_visible_points_count_thrust.data());

    // 2. do the matrix multiplication using cublas.
    // create a vector for matrix reduction
    float *vector_reduction;
    CUDA_CHECK(cudaMallocManaged(&vector_reduction, num_points * sizeof(float)));
    CUDA_CHECK(cudaMemset(vector_reduction, 1, num_points * sizeof(float)));

    float *d_camera_visible_points_count_ptr;
    CUDA_CHECK(cudaMallocManaged(&d_camera_visible_points_count_ptr, num_cameras * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_camera_visible_points_count_ptr, 0, num_cameras * sizeof(float)));

    // create a cublas handle
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    // --- Define scalar multipliers for GEMV ---
    // C = alpha * A * B + beta * C
    float alpha = 1.0f; // Multiply A*B by 1.0
    float beta = 0.0f;  // Overwrite C, don't add to existing C

    // --- Perform Matrix-Vector Multiplication using cublasSgemv ---
    // Measure execution time
    cudaEvent_t start_gemv, stop_gemv;
    CUDA_CHECK(cudaEventCreate(&start_gemv));
    CUDA_CHECK(cudaEventCreate(&stop_gemv));

    printf("\nLaunching cublasSgemv...\n");
    CUDA_CHECK(cudaEventRecord(start_gemv));

    // Perform matrix-vector multiplication
    CUBLAS_CHECK(cublasSgemv(cublasH,
        CUBLAS_OP_N,        // A is row-major visibility matrix, do not transpose
        num_cameras,        // number of rows of op(A)
        num_points,         // number of columns of op(A)
        &alpha,             // scalar multiplier for A*x
        visibility_matrix,  // pointer to matrix A
        num_cameras,        // leading dimension of original A (number of columns in row-major)
        vector_reduction,   // pointer to vector x
        1,                  // stride for vector x (contiguous), 1 means contiguous
        &beta,              // scalar multiplier for beta*C, not used here, thus vector_bias is NULL
        d_camera_visible_points_count_ptr,        // output place
        1));                // stride for vector C (contiguous)

    CUDA_CHECK(cudaEventRecord(stop_gemv));
    CUDA_CHECK(cudaEventSynchronize(stop_gemv));

    float gemv_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gemv_milliseconds, start_gemv, stop_gemv));
    printf("Kernel execution time for matrix-vector multiplication: %f ms\n", gemv_milliseconds);

    // 3. find candidate cameras for final coverage selection
    cudaEvent_t start_candidate, stop_candidate;
    CUDA_CHECK(cudaEventCreate(&start_candidate));
    CUDA_CHECK(cudaEventCreate(&stop_candidate));

    int candidate_blockSize = 256;
    int candidate_gridSize = (num_points + candidate_blockSize - 1) / candidate_blockSize;

    printf("\nStarting CUDA Kernel to find candidate cameras...\n");
    printf("    Grid size: %d, block size: %d\n", candidate_gridSize, candidate_blockSize);

    CUDA_CHECK(cudaEventRecord(start_candidate));
    findCandidateCameraKernel<<<candidate_gridSize, candidate_blockSize>>>(
        visibility_matrix,
        d_camera_visible_points_count_ptr,
        point_candidate_camera_index,
        candidate_camera_mask,
        num_cameras,
        num_points);
    CUDA_CHECK(cudaEventRecord(stop_candidate));
    CUDA_CHECK(cudaEventSynchronize(stop_candidate));

    float candidate_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&candidate_milliseconds, start_candidate, stop_candidate));
    printf("Kernel execution time for finding candidate cameras: %f ms\n", candidate_milliseconds);

    // 4. free the memory
    CUDA_CHECK(cudaFree(vector_reduction));
    CUDA_CHECK(cudaFree(d_camera_visible_points_count_ptr));
    CUBLAS_CHECK(cublasDestroy(cublasH));
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
    float* __restrict__ d_visibility_matrix, // output visibility matrix
    int num_cameras, // number of cameras
    int num_points) { // number of points

    // 1. get camera index and point index
    size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_combinations = (size_t)num_cameras * num_points;

    for (; global_index < total_combinations; global_index += blockDim.x * gridDim.x) {
        int camera_index = global_index / num_points;
        int point_index = global_index % num_points;
        
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
    float* __restrict__ d_visibility_matrix,
    const float* __restrict__ d_camera_visible_points_count,
    int* __restrict__ d_point_candidate_camera_index, // not necessary, actually can be removed.
    unsigned int* __restrict__ d_candidate_camera_mask,
    int num_cameras,
    int num_points) {
    // 1. get point index
    size_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_points = (size_t)num_points;

    // incase the point index is bigger than the number of allocable threads, use the following loop to ensure the point is processed
    for (; point_index < total_points; point_index += blockDim.x * gridDim.x) {
        float max_visible_count_for_point = -1; // stores the maximum numbr of visible points among cameras visible to the current point, -1 as initial value
        int candidate_camera_idx_for_pt = -1; // stores the index of the candidate camera chosen for the current point, -1 means no camera is chosen for the point

        for (int camera_idx = 0; camera_idx < num_cameras; camera_idx++) {
            if (d_visibility_matrix[camera_idx * num_points + point_index] == 1) {
                // the camera is visible to the point, check the camera's visible points count
                float current_camera_visible_count = d_camera_visible_points_count[camera_idx];

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