cuda-video-project
==================

CUDA Video Processing Project

##To build and run parallel black and white

1. make bandw

2. ./bandw

The output frames will be in the outfiles directory

##To build and run parallel 2D convolution

1. make convolution

2. ./convolution

The output frames will be in the outfiles directory

##To build and run the serial black/white and convolution
1. make serial

2. ????????????

##To build and run 3D parallel convolution

1. cd 3Dconvolution

2. make parallel

3. ./par_3D_conv

##To build and run 3D serial convolution

1. cd 3Dconvolution

2. make serial

3. ./ser_3D_conv strideNum

The output file will be in ../outfilter.mp4
ppm takes an optional argument for an input file from ../input_videos/ (default is foreman.mp4)

##To Generate Custom Input Frames:

1. Find a video you want to process

2. Download [FFmpeg](https://ffmpeg.org/)

3. Break the video into frames
  - Use this to break off the first 20 frames of video 'sample_video':
  '''
  ffmpeg -i ./input_videos/sample_video.mp4 -vframes 20 ./infiles/tmp%03d.ppm
  '''

##To Generate Output Video From Frames:

1. Use FFmpeg to combine the output frames into a video:
ffmpeg -framerate 24 -i outfiles/tmp%03d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p output_video.mp4
