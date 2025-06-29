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
    // other mesh related methods
    std::vector<float3>& getVertexes() {
        return vertices_;
    };
private:
    // mesh data members
    std::vector<float3> vertices_; 
};

} // namespace MeshProcessing

#endif // MESH_H 