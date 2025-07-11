#ifndef MESH_H
#define MESH_H

#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace MeshProcessing {
class Mesh {
public:
    Mesh();
    ~Mesh() {};
    void loadFromFile(const std::string& filename);
    void loadFromFile_jetson(const std::string& filename);
    // other mesh related methods
    std::vector<float3>& getVertexes() {
        return vertices_;
    };

public:
    // mesh data members
    std::vector<float3> vertices_; 
    // because the mechanism of rply lib, this is a unavoidable data member that needs to be kept, even in jetson version
    // however, no need to remove it, even in 100w points mesh, the memory usage is around 10MB, which is acceptable

    // for jetson design and usage
    float3* unified_float_vertices_; // allocated in unified memory
    int vertex_num_; // number of vertices, only activated in jetson version
};

} // namespace MeshProcessing

#endif // MESH_H 