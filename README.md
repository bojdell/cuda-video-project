cuda-video-project
==================

CUDA Video Processing Project


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