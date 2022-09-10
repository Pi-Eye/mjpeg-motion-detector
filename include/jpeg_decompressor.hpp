#ifndef JPEG_DECOMPRESSOR_HPP
#define JPEG_DECOMPRESSOR_HPP

#include <turbojpeg.h>

/**
 * DecompVideoFormat - Selector for what format images should be decompressed into
 *
 * kRGB:  < Row: < Pixel: <Red Byte, Green Byte, Blue Byte> ...> ...>
 * kGray: < Row: < Pixel: <Grayscale Byte> ...> ...>
 */
enum class DecompFrameFormat { kRGB, kGray };

/**
 * DecompFrameMethod - Selector for method to use for decompressing images
 *
 * kFast:      faster but less accurate method
 * kAccurate:  slower but more accurate method
 */
enum class DecompFrameMethod { kFast, kAccurate };

/**
 * JpegDecompressor - Decompresses JPEG images into selected format
 */
class JpegDecompressor {
 public:
  /**
   * JpegDecompressor() - Constructor for JpegDecompressor
   *
   * width:         Width of image to decompress
   * height:        Height of image to decompress
   * frame_format:  Format to decompress images into
   * decomp_method: Method to use to decompress images
   */
  JpegDecompressor(unsigned int width, unsigned int height, DecompFrameFormat frame_format, DecompFrameMethod decomp_method);

  /**
   * ~JpegDecompressor - Deconstructor for JpegDecompressor
   */
  ~JpegDecompressor();

  /**
   * DecompressImage() - Decompresses a JPEG image
   *
   * compressed_image:  JPEG image to decompress
   * jpeg_size:         Size of JPEG image to decompress in bytes
   * returns:           unsigned char* - char array of decompressed image
   */
  unsigned char* DecompressImage(unsigned char* compressed_image, unsigned long jpeg_size) const;

  /**
   * GetDecompressedSize() - Get the size of a decompressed image
   *
   * returns:   unsigned int - size of decompressed image
   */
  unsigned int GetDecompressedSize() const;

 private:
  /**
   * DestroyDecompressor() - Destroys the decompressor for JpegDecompressor
   */
  void DestroyDecompressor();

  unsigned int width_;              // Width of image
  unsigned int height_;             // Height of image
  unsigned int decompressed_size_;  // Size of decompressed image buffer

  tjhandle tj_decompressor_;  // TurboJPEG image decompressor handle
  TJPF pixel_format_;         // TurboJPEG pixel format to decompress jpeg images into
  int decomp_flags_;          // TurboJPEG flags for decompressor
};

#endif