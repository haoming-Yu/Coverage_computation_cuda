#include "mesh.h"
#include "rply.h"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "cudaUtils.cuh"

namespace MeshProcessing {

Mesh::Mesh() {
    // implementation of constructor
}

struct PLYReaderState {
    Mesh* mesh_ptr;
    long vertex_index;
    long vertex_num;
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);

    switch (index) {
        case 0:
            state_ptr->mesh_ptr->getVertexes()[state_ptr->vertex_index].x = value;
            break;
        case 1:
            state_ptr->mesh_ptr->getVertexes()[state_ptr->vertex_index].y = value;
            break;
        case 2:
            state_ptr->mesh_ptr->getVertexes()[state_ptr->vertex_index].z = value;
            state_ptr->vertex_index++;
            break;
        default:
            break;
    }
    
    return 1;
}

void Mesh::loadFromFile(const std::string& filename) {
    // implementation to load mesh from file
    std::cout << "[PLY format] Loading mesh from " << filename << std::endl;
    // actual loading code, here we use rply to do the specific reading job
    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        std::cout << "[PLY format] Read PLY failed: unable to open file: " << filename << std::endl;
        return;
    }
    if (!ply_read_header(ply_file)) {
        std::cout << "[PLY format] Read PLY failed: unable to parse header." << std::endl;
        ply_close(ply_file);
        return;
    }

    PLYReaderState state_ptr;
    state_ptr.mesh_ptr = this;
    state_ptr.vertex_index = 0;
    state_ptr.vertex_num = \
    ply_set_read_cb(ply_file, "vertex", "x", ReadVertexCallback, &state_ptr, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state_ptr, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state_ptr, 2);
    
    if (state_ptr.vertex_num <= 0) {
        std::cout << "[PLY format] Read PLY failed: number of vertex <= 0." << std::endl;
        ply_close(ply_file);
        return;
    }

    this->vertices_.resize(state_ptr.vertex_num);
    if (!ply_read(ply_file)) {
        std::cout << "[PLY format] Read PLY failed: unable to read file: " << filename << std::endl;
        ply_close(ply_file);
        return;
    }

    ply_close(ply_file);
    std::cout << "[PLY format] PLY loaded, " << state_ptr.vertex_num << " vertices in total." << std::endl;
}   

void Mesh::loadFromFile_jetson(const std::string& filename) {
    // implementation to load mesh from file
    std::cout << "[PLY format] Loading mesh from " << filename << std::endl;
    // actual loading code, here we use rply to do the specific reading job
    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        std::cout << "[PLY format] Read PLY failed: unable to open file: " << filename << std::endl;
        return;
    }
    if (!ply_read_header(ply_file)) {
        std::cout << "[PLY format] Read PLY failed: unable to parse header." << std::endl;
        ply_close(ply_file);
        return;
    }

    PLYReaderState state_ptr;
    state_ptr.mesh_ptr = this;
    state_ptr.vertex_index = 0;
    state_ptr.vertex_num = \
    ply_set_read_cb(ply_file, "vertex", "x", ReadVertexCallback, &state_ptr, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state_ptr, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state_ptr, 2);
    
    if (state_ptr.vertex_num <= 0) {
        std::cout << "[PLY format] Read PLY failed: number of vertex <= 0." << std::endl;
        ply_close(ply_file);
        return;
    }

    this->vertices_.resize(state_ptr.vertex_num);
    if (!ply_read(ply_file)) {
        std::cout << "[PLY format] Read PLY failed: unable to read file: " << filename << std::endl;
        ply_close(ply_file);
        return;
    }
    this->vertex_num_ = state_ptr.vertex_num;
    CUDA_CHECK(cudaMallocManaged(&this->unified_float_vertices_, sizeof(float3) * this->vertex_num_));
    for (int i = 0; i < this->vertex_num_; i++) {
        this->unified_float_vertices_[i].x = this->vertices_[i].x;
        this->unified_float_vertices_[i].y = this->vertices_[i].y;
        this->unified_float_vertices_[i].z = this->vertices_[i].z;
    }

    ply_close(ply_file);
    std::cout << "[PLY format] PLY loaded, " << state_ptr.vertex_num << " vertices in total." << std::endl;
}

} // namespace MeshProcessing 