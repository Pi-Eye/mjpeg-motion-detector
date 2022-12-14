kernel void stabilize_bg_mvt(global unsigned char* bg_frame_to_remove, global unsigned char* mvt_frame_to_remove, global unsigned char* scaled_frame, global float* bg_length,
                             global float* mvt_length, global float* stabilized_background, global float* stabilized_movement, global int* difference_threshold,
                             global bool* difference_frame_) {
  const int loc = get_global_id(0);

  // Calculate the change in average
  const float bg_change = (mvt_frame_to_remove[loc] / bg_length[0]) - (bg_frame_to_remove[loc] / bg_length[0]);
  const float mvt_change = (scaled_frame[loc] / mvt_length[0]) - (mvt_frame_to_remove[loc] / mvt_length[0]);

  // Change average
  stabilized_background[loc] += bg_change;
  stabilized_movement[loc] += mvt_change;

  // Check if the difference is above the threshold
  difference_frame_[loc] = fabs((stabilized_background[loc] - stabilized_movement[loc])) >= difference_threshold[0];
}