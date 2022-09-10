#include "jpeg_decompressor.hpp"

#include <turbojpeg.h>

#include <stdexcept>

JpegDecompressor::JpegDecompressor(unsigned int width, unsigned int height, DecompFrameFormat frame_format, DecompFrameMethod decomp_method) : width_(width), height_(height) {
  // Create decompressor and throw error if it fails
  tj_decompressor_ = tjInitDecompress();
  if (tj_decompressor_ == NULL) throw std::runtime_error("Failed to initialize JPEG decompressor");

  // Set pixel_format_ flag and decompressed_size appropriately based on frame_format
  switch (frame_format) {
    case (DecompFrameFormat::kGray): {
      pixel_format_ = TJPF::TJPF_GRAY;
      decompressed_size_ = (width_ + 1) * (height_ + 1);
      break;
    }
    case (DecompFrameFormat::kRGB):
    default: {
      pixel_format_ = TJPF::TJPF_RGB;
      decompressed_size_ = (width_ + 1) * (height_ + 1) * 3;
      break;
    }
  }

  // Set decomp_flags_ appropriately based on decomp_method
  switch (decomp_method) {
    case (DecompFrameMethod::kFast): {
      decomp_flags_ = TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE;
    }
    case (DecompFrameMethod::kAccurate):
    default: {
      decomp_flags_ = TJFLAG_ACCURATEDCT;
    }
  }
}

JpegDecompressor::~JpegDecompressor() {
  try {
    DestroyDecompressor();
  } catch (...) {
  }
}

unsigned char* JpegDecompressor::DecompressImage(unsigned char* compressed_image, unsigned long jpeg_size) const {
  // Decompress header of JPEG for metadata
  int width;
  int height;
  int jpeg_subsampling;
  int jpeg_colorspace;
  tjDecompressHeader3(tj_decompressor_, compressed_image, jpeg_size, &width, &height, &jpeg_subsampling, &jpeg_colorspace);

  // If height or width don't match, throw error
  if (width != width_) throw std::out_of_range("Width of compressed JPEG image did not match expected value");
  if (height != height_) throw std::out_of_range("Height of compressed JPEG image did not match expected value");

  // Create destination for image
  unsigned char* decompressed_image = new unsigned char[decompressed_size_];

  // Decompress image and throw error if fails
  int pitch = 0;  // bytes per line in destination image, should be 0 for normal decompression
  int success = tjDecompress2(tj_decompressor_, compressed_image, jpeg_size, decompressed_image, width, pitch, height, pixel_format_, decomp_flags_);
  if (success != 0) throw std::runtime_error("Failed to decompresss image");

  return decompressed_image;
}

unsigned int JpegDecompressor::GetDecompressedSize() const { return decompressed_size_; }

void JpegDecompressor::DestroyDecompressor() {
  // Destroy decompressor and throw error if it fails
  int success = tjDestroy(tj_decompressor_);
  if (success != 0) throw std::runtime_error("Failed to destroy JPEG decompressor");
}