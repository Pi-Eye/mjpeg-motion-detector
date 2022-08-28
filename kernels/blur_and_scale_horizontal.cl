kernel void blur_and_scale_horizontal(global const float* gaussian, global const int* gaussian_size, global const int* scale, global const unsigned char* intermediate_scaled,
                                      global const int* width, global const int* scaled_width, global unsigned char* scaled) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= scaled_width[0]) return;

  // Get the x start location of input frame (y is the same since this is just a horizontal scale down)
  const int input_frame_x_start = scale[0] * x;

  float sum = 0;
  // Iterate through the gaussian
  for (int i = 0; i < gaussian_size[0]; i++) {
    // Find corresponding location in input frame
    const int input_frame_x = input_frame_x_start + i;
    // Calculate the location in the buffer this coordinate is
    const int loc = y * width[0] + input_frame_x;

    // Multiply by gaussian
    sum += (intermediate_scaled[loc]) * gaussian[i];
  }

  // Calculate location in scaled buffer of the coordinate
  const int scaled_loc = y * scaled_width[0] + x;
  scaled[scaled_loc] = sum;
}