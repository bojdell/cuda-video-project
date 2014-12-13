CUDA Video Processing Project
========================

This project utilizes parallel computing to improve the efficiency of post-processing  video footage.

The code currently supports post-processing in up to 3 dimensions, using convolution filters.


###How To Process A Video:

1. Find a video you want to process, and put it in `./input_videos`

2. Download and install [FFmpeg](https://ffmpeg.org/)

3. Use FFmpeg to break the video into frames. This code breaks off the first 20 frames of video `sample_video`:

  ```
  ffmpeg -i input_videos/sample_video.mp4 -vframes 20 infiles/tmp%03d.ppm
  ```

4. Choose an operation, compile and run it

  - Serial (slower execution)
     *  1D Convolution (Black & White Recoloring): `make serial-bw`, then `./serial-bw`
     *  2D Convolution: `make serial-convolution`, then `./serial-convolution`
     *  3D Convolution: `cd 3Dconvolution`, `make serial`, then `./ser_3D_conv`
  - Parallel (faster execution, requires [CUDA](https://developer.nvidia.com/cuda-downloads) and a valid GPU)
     *  1D Convolution (Black & White Recoloring): `make bandw`, then `./bandw`
     *  2D Convolution: `make convolution `, then `./convolution`
     *  3D Convolution: `cd 3Dconvolution`, `make parallel`, then `./par_3D_conv`

  Where `stride_len` is an optional parameter that can be used to control the number of frames stored in memory at one time. The default value is 20.

5. Use FFmpeg to combine the output frames into a video:
```
ffmpeg -framerate 24 -i outfiles/tmp%03d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p output_video.mp4
```