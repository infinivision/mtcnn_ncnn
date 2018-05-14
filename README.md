# MTCNN NCNN Implementation
-----------------------------
the application can be run on X86 and ARM platform.

# How to build
## clone the repo
```sh
git clone --recursive https://github.com/infinivision/mtcnn_ncnn.git
```

## build ncnn
* ncnn as submodule for the main repo
* to support mtcnn. I update the ncnn code. [the repo addr](https://github.com/infinivision/ncnn.git) 
* for the build and install please refer to the ncnn [wiki page](https://github.com/Tencent/ncnn/wiki/how-to-build)

## copy the ncnn lib and include headers to the main repo folder
```sh
# by default the ncnn was installed in ncnn/build/install folder
mkdir -p lib/ncnn
mkdir -p include/ncnn
cp 3rdparty/ncnn/build/install/lib/libncnn.a lib/ncnn/
3rdparty/ncnn/build/install/include/* include/ncnn/
```

## build the mtcnn main repo
```sh
mkdir -p build
cd build
cmake ..
make
```

# how to run
## test picture
```sh
cd bin
./test_picture ../models/ncnn/ ../images/1.jpg
```
## test video
```sh
cd bin
./test_video ../models/ncnn 0
```

# benchmark
```sh
# we use the [WIDER FACE validate dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) as the test data
cd bin
./benchmark ../models/ncnn ../images
```
## test result on mac os x86 platform

## test result on firefly 3399 ARM CPU(without NEON optimation)

## test result on firefly 3399 ARM CPU(with NEON optimation)

# known issues
* ARM NEON optimation still not work