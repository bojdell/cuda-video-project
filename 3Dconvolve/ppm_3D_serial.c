#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "ppm.h"

void freeImage(PPMImage * image) {
    if(image) {
        if(image->data)
            free(image->data);
        free(image);
    }
}

static PPMImage *readPPM(const char *filename)
{
     char buff[16];
     PPMImage *img;
     FILE *fp;
     int c, rgb_comp_color;
     //open PPM file for reading
     fp = fopen(filename, "rb");
     if (!fp) {
          // fprintf(stderr, "Hey! Unable to open file '%s'\n", filename);
          // exit(1);
        return NULL;
     }

     //read image format
     if (!fgets(buff, sizeof(buff), fp)) {
          perror(filename);
          exit(1);
     }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

//       process
//       ..... -----
// 0 0 0 1 1 1 1 1 1
void filterPPM_3D(PPMImage *frames[], int num_frames, Filter3D f) {
    if(!frames) {
        return;
    }

    int frame;
    int img_x, img_y;
    int f_x = 1, f_y = 1, f_z = 3;
    int pixel_x, pixel_y;
    int frame_offset = f_z; // offset start so we have previous images to convolve with

    for(frame = frame_offset; frame < f_z; frame++) {


    }

}

// read n frames into images[], starting at frame offset
void readFrames(PPMImage ** images, int n, int offset) {
    int j;
    for(j = 0; j < n; j++) {
        sprintf(instr, "infiles/baby%03d.ppm", j + 1);
        PPMImage * img = readPPM(instr);
        if(img == NULL) {
            printf("All files processed\n");
            return;
        }
        images[j + offset] = img;
    }
}

double processImage(PPMImage ** images, Filter3D f, cdim3 stride) {
    if(!images || !images[0]) {
        return -1;
    }

    double time_spent = -1.0;
    clock_t begin, end;

    char instr[80];         // filename for input file
    char outstr[80];        // filename for output file

    int numFrames = 301;    // max # frames

    // calculate number of chunks in each dimension
    cdim3 numChunks = {
        (int)ceil((double)images[0]->x / stride.x),
        (int)ceil((double)images[0]->y / stride.y),
        (int)ceil((double)numFrames / stride.z)
    };

    // loop vars
    int i = 0, j = 0, k = 0;

    // for each plane of input chunks
    for(i = 0; i < numFrames + stride.z - 1; i += stride.z) {

        // read stride.z frames into images[], with f.z / 2 padding in front and back
        if()
        readFrames(images, stride.z + f.z / 2, f.z / 2)
        
        // process chunks of size stride.x * stride.y in images[]
        for(j = 0; j < stride_len && i + j < numFrames; j++) {
        //for(j = 0; j < 1; j++) {

            PPMImage * img = images[j];
            
            begin = clock();
            // filterPPM_3D(img, f);
            end = clock();
            time_spent += ((double)(end - begin) / CLOCKS_PER_SEC);
        }
        
        

        // write 1 chunk of stride_len frames from images[]
        for(j = 0; j < stride_len && i + j < numFrames; j++) {
            sprintf(outstr, "outfiles/baby%03d.ppm", i + j + 1);
            writePPM(outstr, images[j]);
            freeImage(images[j]);
        }
    }

    return time_spent;
}

// ppm <stride_len> (<max_frames>)
int main(int argc, char *argv[]) {
    // argc should be at least 2 for correct execution
    if (argc < 2)
    {
        // We print argv[0] assuming it is the program name
        printf("usage: %s <stride_len> (<max_frames>)\n", argv[0]);
        return -1;
    }

    int stride_len = atoi(argv[1]);

    clock_t begin, end;
    double time_spent;   // time spent for each chunk, plus accumulate total at end

    PPMImage * images[stride_len];
    Filter3D nothing = {
        .x = 5,
        .y = 5,
        .z = 5,
        .data =    { { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 1, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} },

                     { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 1, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} },

                     { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 1, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} },

                     { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 1, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} },

                     { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 1, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} }
                   },
        .factor = 1.0,
        .bias = 0
    };

    // set the stride we will use to break up our files
    cdim3 stride = { 10, 10, stride_len };

    // loop over all frames, in chunks of stride_len
    begin = clock();

    double calc_time = processImage(images, nothing, stride);

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("%f seconds spent\n%f seconds spent processing\n", time_spent, calc_time);

    return 0;
}
