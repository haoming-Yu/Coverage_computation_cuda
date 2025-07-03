# CUDA Coverage 
**Hello! Welcome to my repo of using cuda to increase the speed for camera mesh coverage computation, here are something you should know about this repo before use:** 

## What is this repo for
during my usage of mvs-texture(https://github.com/nmoehrle/mvs-texturing)，I found that the captured images might have two problems which cause the result to be bad:

1. the images might be too sparse to cover the whole scene. Thus I need to capture images at a high frequency. In my case, I set the storage frequence at 1 FPS.

2. However, if the images are captured or filtered at a fixed frequency using a scanner(which is a mostly possible design for robotics, scanner, and many SLAM scenarios), it would be hard to control the images cover the whole scene while use least images to do that(reduce the redundant images).

Based on those observations, I decide to write a coverage post-processing node to control the image number. In other word, reduce the redundant image frames while maximize the coverage of the scene.

I write a CPU to test whether the idea is possible.(https://github.com/haoming-Yu/cpu_projection_timing_test) However, on Intel i9 14900K(almost the most powerful CPU on PC in 2024), it took me about 12 seconds to do the coordinate transformation from world to camera. (Single core, even if I use powerful TBB and OpenMP, the constrained number of cpu core can not process the simple operation within reasonable time when the scene becomes bigger. For more analyse and runtime test, you can check this doc. (For now it is written in Chinese): https://www.yuque.com/yuhaominghyu/vlmadq/gkwr3yvkcheqggc9?singleDoc# 《Record on Developing CUDA speeding module》)

That's why I decide to develop the CUDA version to harness the potential of large scale parallel capability on NVIDIA GPUs.

## Problem Description
### Input
1. a triangle mesh in PLY format.
2. a folder contains images.
3. a folder contains depth maps corresponding to the images. (In my case, I use OpenGL to render the mesh to camera images to get the depth maps)
4. a trajectory file contains the images' extrinsic matrix. (camera to world)
5. a file contains the images' and depth maps' intrinsic information.

### Output
a vector with length of camera images, each has a value of 1 and 0. If it is 1, it indicates that the camera image with corresponding index is selected.

## Installation and building
You just need to ensure you have `Eigen`, `OpenCV`, `CUDA`, and `CUDA thrust lib` installed.

Then you can start build the project. If there are problems, remember to check the CMakeLists.txt, and ensure all the libs required can be correctly found. In my case, I use Ubuntu 20.04 to do the test with nvcc 11.8.

First Enter the main directory of this project, then follow the instructions below. Remember to prepare the dataset before running.

```shell
mkdir build && cd build
cmake ..
make
cd ..
# now check the dataset and the instruction inside launch.sh
./launch.sh
```
## Result
1. On NVIDIA RTX 4090 GPU, I can achieve to make it less than 5ms to process the problem. And I intend to first check mvs-texturing's correctness, then adjust the code to support Jetson platform.
2. The Jetson has been tested, and it works well. You can switch to jetson branch to find the code with a few slight changes on the CMakeLists.txt and Launch. For now, the cuda is not changed. And might explore other cuda memory result on jetson platform. For now, cuda memory copy seems to perform a zero copy but conversion inside unified memory operation which surprise me.
3. tested, cuda memory copy will increase the usage of swap memory, so need to be optimized to use less memory.
## What's more
The repo is still under development, and I am the only developer to  maintain this repo. Any problem and issue is welcomed! But it may take some time for me to fix it. I am still a greenhorn on CUDA, and a first-year PhD student. Looking forward to anyone's precious suggestion.

Thanks for your time.