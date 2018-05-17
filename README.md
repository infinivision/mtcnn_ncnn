# MTCNN NCNN Implementation
-----------------------------
the application can be run on X86 and ARM platform with Neon optimization.

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
cp 3rdparty/ncnn/build/install/include/* include/ncnn/
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
## Test data
we use the [WIDER FACE validate dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) as the test data

## run benchmark
```sh
cd bin
./benchmark ../models/ncnn ../images
```

## Test Result
| Platform   | CPU Cores |  Memory  |  total images |   detected   |  min time(ms) | max time(ms) | avg time(ms) |
| --------   | :-----:    | :----:   | :----:        |    :----:    |    :----:  | :----:    | :----:    |
| MacOS Sierra | 8    |   16GB  |   3226     |  2073     |   8.068    |   287.446    |   47.03      |
| firefly 3399 | 6    |   2GB     |   3226       |   2084      |   25.541     |   4437.42    |    351.578    |
| firefly 3399(Neon) | 6    |   2GB     |   3226       |   2084      |   10.578     |   1325.09    |   157.908   |

