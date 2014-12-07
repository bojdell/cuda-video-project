#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "ppm.h"

void freeImage(PPMImage ** image) {
    if(*image) {
        if((*image)->data) {
            free((*image)->data);
            (*image)->data = NULL;
        }
        free(*image);
        *image = NULL;
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
void filterPPM_3D(PPMImage **images, int idx, Filter3D f) {
    if(!images) {
        return;
    }

    int img_x, img_y;
    int f_x, f_y, f_z;
    int pixel_x, pixel_y, pixel_z;
    int red = 0, green = 0, blue = 0;
    PPMImage * img = images[idx];
    PPMPixel * new_data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    // for each pixel in the image
    for(img_x = 0; img_x < img->x; img_x++) {
        for(img_y = 0; img_y < img->y; img_y++) {

            // apply the filter to surrounding pixels
            for(f_x = 0; f_x < f.x; f_x++) {
                for(f_y = 0; f_y < f.y; f_y++) {
                    for(f_z = 0; f_z < f.z; f_z++) {
                        // get absolute locations of pixel
                        pixel_x = img_x + (f_x - f.x / 2);
                        pixel_y = img_y + (f_y - f.y / 2);
                        pixel_z = idx + (f_z - f.z / 2);

                        // check to make sure coordinates are vaild (don't need to check z b/c of input/output tile relation)
                        if(pixel_x >= 0 && pixel_x < img->x && pixel_y >= 0 && pixel_y < img->y) {
                            int curr_red = images[pixel_z]->data[pixel_y * img->x + pixel_x].red;
                            int curr_green = images[pixel_z]->data[pixel_y * img->x + pixel_x].green;
                            int curr_blue = images[pixel_z]->data[pixel_y * img->x + pixel_x].blue;
                            int fval = f.data[f_z][f_y][f_x];

                            red += curr_red * fval;
                            green += curr_green * fval;
                            blue += curr_blue * fval;
                        }
                    }
                }
            }

            // apply filter factor and bias, and write resultant pixel to new_data
            new_data[img_y * img->x + img_x].red = fmin( fmax( (int)(f.factor * red + f.bias), 0), 255);
            new_data[img_y * img->x + img_x].green = fmin( fmax( (int)(f.factor * green + f.bias), 0), 255);
            new_data[img_y * img->x + img_x].blue = fmin( fmax( (int)(f.factor * blue + f.bias), 0), 255);
        }
    }

    free(img->data);
    img->data = new_data;
}

// read n frames into images[], starting at frame start. if start < 0, leave pointers null
void readFrames(PPMImage ** images, int num_images, int n, int start) {
    char instr[80]; // filename for input file

    int j;
    for(j = 0; j < n; j++) {
        if(j + start >= 0 && j + start < num_images) {
            sprintf(instr, "infiles/baby%03d.ppm", j + start);
            PPMImage * img = readPPM(instr);
            if(!img) {
                return;
            }
            images[j + start] = img;
        }
        else {
            // if(images[j + start])
                // freeImage(&images[j + start]);
        }
    }
}

// write n frames from images[] to ppm files, unless pointer is null
void writeFrames(PPMImage ** images, int num_images, int n, int start) {
    char outstr[80];    // filename for output file

    int j;
    for(j = 0; j < n; j++) {
        if(j + start >= 0 && j + start < num_images && images[j + start]) {
            sprintf(outstr, "outfiles/baby%03d.ppm", j + start);
            writePPM(outstr, images[j + start]);
            // freeImage(&images[j]);
        }
    }
}

// void processChunk(PPMImage * image, int start_x, int start_y, int end_x, int end_y) {
//     if(!image || start_x > end_x || start_y > end_y)
//         return;

//     int i, j;
//     for(i = start_x; i <= end_x; x++) {
//         for(j = start_y; j <= end_y; y++) {
//             image->data[]
//         }
//     }
// }

double processImages(PPMImage ** images, Filter3D f, cdim3 stride) {
    if(!images) {
        return -1;
    }

    double time_spent = -1.0;
    clock_t begin, end;

    int numFrames = 20;    // max # frames

    // calculate number of chunks in each dimension
    int numChunks = (int)ceil((double)numFrames / stride.z);
    // cdim3 numChunks = {
    //     (int)ceil((double)images[0]->x / stride.x),
    //     (int)ceil((double)images[0]->y / stride.y),
    //     (int)ceil((double)numFrames / stride.z)
    // };

    // loop vars
    int i = 0, j = 0;

    // for each plane of input chunks
    for(i = 0; i < numChunks; i += stride.z) {

        // read stride.z frames into images[], with f.z / 2 padding in front and back
        readFrames(images, numFrames, stride.z, i - f.z / 2);

        // process each frame we want to output (i.e. i + f.z/2 to i + stride.z - f.z/2)
        for(j = f.z / 2; j < stride.z - f.z / 2; j++) {
            if(!images[i + j]) {
                break;
            }
            filterPPM_3D(images, i + j, f);
        }
        
        // process chunks of size stride.x * stride.y in images[]
        // for(j = 0; j < numChunks.y; j++) {
        //     for(k = 0; k < numChunks.x; k++) {
        //         PPMImage * img = images[j];
                
        //         begin = clock();
        //         // filterPPM_3D(img, f);
        //         end = clock();
        //         time_spent += ((double)(end - begin) / CLOCKS_PER_SEC);
        //     }
        // }
        
        // write stride.z frames from images[] to ppm files
        writeFrames(images, numFrames, stride.z, i - f.z / 2);
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

    // read in stride_len (stride in z-direction)
    int stride_len = atoi(argv[1]);

    clock_t begin, end;
    double time_spent;

    Filter3D f = {
        .x = 5,
        .y = 5,
        .z = 5,
        .data =    { { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} },

                     { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} },

                     { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 1, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} },

                     { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} },

                     { {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0} }
                   },
        .factor = 1.0,
        .bias = 0
    };

    // create space for video frames (including padding)
    PPMImage * images[stride_len + f.z - 1];

    // set the stride we will use to break up our files
    cdim3 stride = { 10, 10, stride_len + f.z - 1 };

    begin = clock();
    printf("1\n");
    // loop over all frames, in chunks of stride_len
    double calc_time = processImages(images, f, stride);

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("%f seconds spent\n%f seconds spent processing\n", time_spent, calc_time);

    return 0;
}
