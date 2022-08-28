kernel void blur_and_scale_vertical(global const float* gaussian, global const int* gaussian_size, global const int* scale, global const int* colors,
                                    global const unsigned char* frame, global const int* width, global unsigned char* scaled) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= width[0]) return;

  // Get the y start location of input frame (x is the same since this is just a vertical scale down)
  const int input_frame_y_start = scale[0] * y;

  float sum = 0;
  // Iterate through the gaussian
  for (int i = 0; i < gaussian_size[0]; i++) {
    // Find corresponding location in input frame
    const int input_frame_y = input_frame_y_start + i;
    // Calculate the location in the buffer this coordinate is
    const int loc = (input_frame_y * width[0] + x) * colors[0];

    // Add up all the colors
    int color_total = 0;
    for (int c = 0; c < colors[0]; c++) {
      color_total += frame[loc + c];
    }

    // Multiply by gaussian
    sum += color_total * gaussian[i];
  }

  // Calculate location in scaled buffer of the coordinate
  const int scaled_loc = y * width[0] + x;
  scaled[scaled_loc] = sum / colors[0];  // Divide by the number of colors to normalize
}