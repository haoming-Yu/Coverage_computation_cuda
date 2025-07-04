# Becareful when you use this script, this script is used in my own environment to test mvs-texture's result. 
# If you want to use the script, please make sure three things:
# 1. the data_path is the path to the data, and the data is carefully prepared
# 2. the mvs-texture is built, and set the path to the mvs-texture in the script
# 3. the cuda_coverage is built and run successfully
export data_path="/home/jetson/Coverage_computation_cuda/data/underground_lighter"
echo "data_path: $data_path"
# run a coverage GPU program
./build/COVERAGE -m $data_path/mesh/filtered_mesh.ply -i $data_path/intrinsic.log -e $data_path/traj.log -d $data_path/depth -p $data_path/image
# create a workspace for mvs-texture to put the result
cd $data_path
# if tex_obj exists, delete it together with all files in it
if [ -d "tex_obj" ]; then
    rm -rf tex_obj
fi
mkdir -p tex_obj/result
# run mvs-texture
/home/jetson/mvs-texturing/build/apps/texrecon/texrecon \
    --smoothness_term=potts \
    --tone_mapping=gamma \
    --outlier_removal=gauss_damping \
    --keep_unseen_faces \
    --num_threads=5 \
    $data_path/image_filtered_with_cam \
    $data_path/mesh/filtered_mesh.ply \
    $data_path/tex_obj/result