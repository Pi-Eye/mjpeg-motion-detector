#include "jpeg_decompressor.hpp"

JpegDecompressor::JpegDecompressor(unsigned int width, unsigned int height, DecompFrameFormat frame_format, DecompFrameMethod decomp_method) {}

JpegDecompressor::~JpegDecompressor() {}

unsigned char* JpegDecompressor::DecompressImage(unsigned char* compressed_image, unsigned long int jpeg_size) const {}

unsigned int JpegDecompressor::GetDecompressedSize() const { return decompressed_size_; }

void JpegDecompressor::DestroyDecompressor() {}