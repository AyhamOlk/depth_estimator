# Computer Vision Disparity calculation
This program discusses a pipeline to find the disparity map of two stereo images that contain specific objects, which are cars in this case. In this work, we build a pipeline that is based on previous well studied and known techniques and algorithms. Our pipeline consists of an Object-detection layer followed by an images matching box that feeds the resulting good matches to an 8-point algorithm that approximates the Fundamental matrix of the camera used in this work; at the end we build our disparity map over multiple correlation metrics between images like Zero Mean Normalized Cross Correlation (ZNCC), Sum of Absolute Differences (SAD) and Sum of Square Differences (SSD). This work could be used for autonomous car projects that tries to estimate depth of objects in images. For more details, refer to our report in the repository.

# Requirements
For the requirements refer to the file *requirements.txt*.

# Installation and execution
Clone the repo using:
```
git clone https://github.com/AyhamOlk/depth_estimator.git
```
which will create a folder called *depth_estimator*. Then run the following commands in sequence:
```
cd depth_estimator
mkdir build
cd build
cmake ..
make
```
After these commands the following exectuable files will be created: approximate_F and disparity.
To be able to achieve the depth map for certain images, make sure to place the correct paths in the **detect.py** and **disparity.cpp**. Finally the whole story is about typing the following commands:
```
python3 detect.py
./approximate_F
./disparity
```
The resulting depth map image will be located in the *runs/disparity* folder under the name **disparity.jpg**.

In case of facing issues, do not hesitate to contact us:
Ayham Olleik <ayham.olleik@polytechnique.edu>
Khalig Aghakarimov <khalig.aghakarimov@polytechnique.edu>
