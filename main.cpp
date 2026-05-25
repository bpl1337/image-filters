#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <immintrin.h>
#include <omp.h>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

using Byte = std::uint8_t;
using Image = std::vector<Byte>;  

struct ImageView {
  int width;
  int height;
  int channels;  
  Image pixels;
};

ImageView LoadPng(const std::string& path) {
  int width = 0;
  int height = 0;
  int channels_in_file = 0;
  Byte* raw = stbi_load(path.c_str(), &width, &height, &channels_in_file, 3);
  if (raw == nullptr) {
    throw std::runtime_error("stbi_load failed: " + path);
  }
  ImageView image{width, height, 3,
                  Image(raw, raw + static_cast<std::size_t>(width) * height * 3)};
  stbi_image_free(raw);
  return image;
}

void SavePng(const std::string& path, const ImageView& image) {
  const int stride = image.width * image.channels;
  if (!stbi_write_png(path.c_str(), image.width, image.height, image.channels,
                      image.pixels.data(), stride)) {
    throw std::runtime_error("stbi_write_png failed: " + path);
  }
}

inline Byte ClampToByte(int value) {
  return static_cast<Byte>(std::min(255, std::max(0, value)));
}

ImageView InvertSequential(const ImageView& input) {
  Image output(input.pixels.size());
  std::transform(input.pixels.cbegin(), input.pixels.cend(), output.begin(),
                 [](Byte pixel) { return static_cast<Byte>(255 - pixel); });
  return {input.width, input.height, input.channels, std::move(output)};
}

ImageView InvertOpenMp(const ImageView& input) {
  Image output(input.pixels.size());
  const auto count = static_cast<std::int64_t>(input.pixels.size());
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < count; ++i) {
    output[i] = static_cast<Byte>(255 - input.pixels[i]);
  }
  return {input.width, input.height, input.channels, std::move(output)};
}

ImageView InvertSimd(const ImageView& input) {
  Image output(input.pixels.size());
  const std::size_t count = input.pixels.size();
  const std::size_t step = 32;  
  std::size_t i = 0;
  const __m256i ones = _mm256_set1_epi8(static_cast<char>(0xFF));
  for (; i + step <= count; i += step) {
    __m256i pixels = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(input.pixels.data() + i));
    __m256i inverted = _mm256_xor_si256(pixels, ones);  
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output.data() + i), inverted);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<Byte>(255 - input.pixels[i]);
  }
  return {input.width, input.height, input.channels, std::move(output)};
}

inline Byte Median9(std::array<Byte, 9> values) {
  std::nth_element(values.begin(), values.begin() + 4, values.end());
  return values[4];
}

ImageView MedianSequential(const ImageView& input) {
  const int width = input.width;
  const int height = input.height;
  const int channels = input.channels;
  Image output(input.pixels.size());

  auto pixel_at = [&](int x, int y, int c) {
    const int xx = std::min(std::max(x, 0), width - 1);
    const int yy = std::min(std::max(y, 0), height - 1);
    return input.pixels[(yy * width + xx) * channels + c];
  };

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        std::array<Byte, 9> neighborhood{};
        int k = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            neighborhood[k++] = pixel_at(x + dx, y + dy, c);
          }
        }
        output[(y * width + x) * channels + c] = Median9(neighborhood);
      }
    }
  }
  return {width, height, channels, std::move(output)};
}

ImageView MedianOpenMp(const ImageView& input) {
  const int width = input.width;
  const int height = input.height;
  const int channels = input.channels;
  Image output(input.pixels.size());

#pragma omp parallel for schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int y_up = std::max(y - 1, 0);
      const int y_down = std::min(y + 1, height - 1);
      const int x_left = std::max(x - 1, 0);
      const int x_right = std::min(x + 1, width - 1);
      for (int c = 0; c < channels; ++c) {
        std::array<Byte, 9> neighborhood{
            input.pixels[(y_up * width + x_left) * channels + c],
            input.pixels[(y_up * width + x) * channels + c],
            input.pixels[(y_up * width + x_right) * channels + c],
            input.pixels[(y * width + x_left) * channels + c],
            input.pixels[(y * width + x) * channels + c],
            input.pixels[(y * width + x_right) * channels + c],
            input.pixels[(y_down * width + x_left) * channels + c],
            input.pixels[(y_down * width + x) * channels + c],
            input.pixels[(y_down * width + x_right) * channels + c]};
        output[(y * width + x) * channels + c] = Median9(neighborhood);
      }
    }
  }
  return {width, height, channels, std::move(output)};
}

static inline void Vec9Median(__m256i& a, __m256i& b, __m256i& c, __m256i& d,
                              __m256i& e, __m256i& f, __m256i& g, __m256i& h,
                              __m256i& i) {
  auto swap_min_max = [](__m256i& x, __m256i& y) {
    __m256i lo = _mm256_min_epu8(x, y);
    __m256i hi = _mm256_max_epu8(x, y);
    x = lo;
    y = hi;
  };
  swap_min_max(a, b); swap_min_max(d, e); swap_min_max(g, h);
  swap_min_max(b, c); swap_min_max(e, f); swap_min_max(h, i);
  swap_min_max(a, b); swap_min_max(d, e); swap_min_max(g, h);
  swap_min_max(a, d); swap_min_max(b, e); swap_min_max(c, f);
  swap_min_max(d, g); swap_min_max(e, h); swap_min_max(f, i);
  swap_min_max(a, d); swap_min_max(b, e); swap_min_max(c, f);
  swap_min_max(b, d); swap_min_max(c, e); swap_min_max(f, h);
  swap_min_max(c, d); swap_min_max(e, f);
}

ImageView MedianSimd(const ImageView& input) {
  const int width = input.width;
  const int height = input.height;
  const int channels = input.channels;
  const std::size_t plane_size = static_cast<std::size_t>(width) * height;
  std::vector<Image> planes(channels, Image(plane_size));
  std::vector<Image> result_planes(channels, Image(plane_size));

  for (int c = 0; c < channels; ++c) {
    for (std::size_t p = 0; p < plane_size; ++p) {
      planes[c][p] = input.pixels[p * channels + c];
    }
  }

  auto sample = [&](const Image& plane, int x, int y) {
    const int xx = std::min(std::max(x, 0), width - 1);
    const int yy = std::min(std::max(y, 0), height - 1);
    return plane[yy * width + xx];
  };

  for (int c = 0; c < channels; ++c) {
    const Image& src = planes[c];
    Image& dst = result_planes[c];
    for (int y = 0; y < height; ++y) {
      int x = 0;
      if (y > 0 && y < height - 1) {
        const Byte* row_up = src.data() + (y - 1) * width;
        const Byte* row_mid = src.data() + y * width;
        const Byte* row_down = src.data() + (y + 1) * width;
        x = 1;
        for (; x + 32 + 1 <= width; x += 32) {
          __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_up + x - 1));
          __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_up + x));
          __m256i cc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_up + x + 1));
          __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_mid + x - 1));
          __m256i e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_mid + x));
          __m256i f = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_mid + x + 1));
          __m256i g = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_down + x - 1));
          __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_down + x));
          __m256i ii = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_down + x + 1));
          Vec9Median(a, b, cc, d, e, f, g, h, ii);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data() + y * width + x), e);
        }
      }
      for (; x < width; ++x) {
        std::array<Byte, 9> nb{
            sample(src, x - 1, y - 1), sample(src, x, y - 1), sample(src, x + 1, y - 1),
            sample(src, x - 1, y),     sample(src, x, y),     sample(src, x + 1, y),
            sample(src, x - 1, y + 1), sample(src, x, y + 1), sample(src, x + 1, y + 1)};
        dst[y * width + x] = Median9(nb);
      }
    }
  }

  Image output(input.pixels.size());
  for (int c = 0; c < channels; ++c) {
    for (std::size_t p = 0; p < plane_size; ++p) {
      output[p * channels + c] = result_planes[c][p];
    }
  }
  return {width, height, channels, std::move(output)};
}

inline Byte Luma(Byte r, Byte g, Byte b) {
  return static_cast<Byte>((r * 30 + g * 59 + b * 11) / 100);
}

static Image RgbToLuma(const ImageView& image) {
  Image luma(static_cast<std::size_t>(image.width) * image.height);
  for (std::size_t p = 0, n = luma.size(); p < n; ++p) {
    luma[p] = Luma(image.pixels[p * 3], image.pixels[p * 3 + 1],
                   image.pixels[p * 3 + 2]);
  }
  return luma;
}

ImageView EdgeSequential(const ImageView& input) {
  const int width = input.width;
  const int height = input.height;
  const Image luma = RgbToLuma(input);
  Image output(input.pixels.size(), 0);

  auto at = [&](int x, int y) {
    return luma[std::min(std::max(y, 0), height - 1) * width +
                std::min(std::max(x, 0), width - 1)];
  };

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int gx = -at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1)
                   +  at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1);
      const int gy = -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1)
                   +  at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1);
      const int magnitude = std::abs(gx) + std::abs(gy);
      const Byte v = ClampToByte(magnitude);
      const std::size_t base = (static_cast<std::size_t>(y) * width + x) * 3;
      output[base] = output[base + 1] = output[base + 2] = v;
    }
  }
  return {width, height, input.channels, std::move(output)};
}

ImageView EdgeOpenMp(const ImageView& input) {
  const int width = input.width;
  const int height = input.height;
  const Image luma = RgbToLuma(input);
  Image output(input.pixels.size(), 0);

#pragma omp parallel for schedule(static)
  for (int y = 0; y < height; ++y) {
    const int y_up = std::max(y - 1, 0);
    const int y_down = std::min(y + 1, height - 1);
    for (int x = 0; x < width; ++x) {
      const int x_left = std::max(x - 1, 0);
      const int x_right = std::min(x + 1, width - 1);
      const int gx = -luma[y_up * width + x_left] - 2 * luma[y * width + x_left]
                     - luma[y_down * width + x_left]
                     + luma[y_up * width + x_right] + 2 * luma[y * width + x_right]
                     + luma[y_down * width + x_right];
      const int gy = -luma[y_up * width + x_left] - 2 * luma[y_up * width + x]
                     - luma[y_up * width + x_right]
                     + luma[y_down * width + x_left] + 2 * luma[y_down * width + x]
                     + luma[y_down * width + x_right];
      const Byte v = ClampToByte(std::abs(gx) + std::abs(gy));
      const std::size_t base = (static_cast<std::size_t>(y) * width + x) * 3;
      output[base] = output[base + 1] = output[base + 2] = v;
    }
  }
  return {width, height, input.channels, std::move(output)};
}

ImageView EdgeSimd(const ImageView& input) {
  const int width = input.width;
  const int height = input.height;
  const Image luma = RgbToLuma(input);
  Image output(input.pixels.size(), 0);

  for (int y = 1; y < height - 1; ++y) {
    const Byte* row_up = luma.data() + (y - 1) * width;
    const Byte* row_mid = luma.data() + y * width;
    const Byte* row_down = luma.data() + (y + 1) * width;
    int x = 1;
    for (; x + 16 + 1 <= width; x += 16) {
      auto load16 = [](const Byte* ptr) {
        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
        return _mm256_cvtepu8_epi16(raw);
      };
      __m256i a = load16(row_up + x - 1);
      __m256i b = load16(row_up + x);
      __m256i c = load16(row_up + x + 1);
      __m256i d = load16(row_mid + x - 1);
      __m256i f = load16(row_mid + x + 1);
      __m256i g = load16(row_down + x - 1);
      __m256i h = load16(row_down + x);
      __m256i i = load16(row_down + x + 1);

      __m256i two = _mm256_set1_epi16(2);
      __m256i gx_pos = _mm256_add_epi16(_mm256_add_epi16(c, i),
                                        _mm256_mullo_epi16(f, two));
      __m256i gx_neg = _mm256_add_epi16(_mm256_add_epi16(a, g),
                                        _mm256_mullo_epi16(d, two));
      __m256i gx = _mm256_sub_epi16(gx_pos, gx_neg);

      __m256i gy_pos = _mm256_add_epi16(_mm256_add_epi16(g, i),
                                        _mm256_mullo_epi16(h, two));
      __m256i gy_neg = _mm256_add_epi16(_mm256_add_epi16(a, c),
                                        _mm256_mullo_epi16(b, two));
      __m256i gy = _mm256_sub_epi16(gy_pos, gy_neg);

      __m256i magnitude = _mm256_add_epi16(_mm256_abs_epi16(gx),
                                           _mm256_abs_epi16(gy));
      __m256i packed = _mm256_packus_epi16(magnitude, magnitude);
      __m256i permuted = _mm256_permute4x64_epi64(packed, 0b11011000);
      alignas(32) Byte temp[32];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), permuted);
      for (int k = 0; k < 16; ++k) {
        const Byte v = temp[k];
        const std::size_t base = (static_cast<std::size_t>(y) * width + x + k) * 3;
        output[base] = output[base + 1] = output[base + 2] = v;
      }
    }
    auto at = [&](int xx, int yy) {
      return luma[std::min(std::max(yy, 0), height - 1) * width +
                  std::min(std::max(xx, 0), width - 1)];
    };
    for (; x < width; ++x) {
      const int gx = -at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1)
                   +  at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1);
      const int gy = -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1)
                   +  at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1);
      const Byte v = ClampToByte(std::abs(gx) + std::abs(gy));
      const std::size_t base = (static_cast<std::size_t>(y) * width + x) * 3;
      output[base] = output[base + 1] = output[base + 2] = v;
    }
  }
  return {width, height, input.channels, std::move(output)};
}

static const char* kOpenClSource = R"CLC(
__kernel void invert(__global const uchar* input, __global uchar* output,
                     const int total) {
  int i = get_global_id(0);
  if (i < total) output[i] = (uchar)(255 - input[i]);
}

inline void sort2(uchar* a, uchar* b) {
  uchar lo = min(*a, *b);
  uchar hi = max(*a, *b);
  *a = lo; *b = hi;
}

__kernel void median3x3(__global const uchar* input, __global uchar* output,
                        const int width, const int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;
  for (int c = 0; c < 3; ++c) {
    uchar v[9];
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int xx = clamp(x + dx, 0, width - 1);
        int yy = clamp(y + dy, 0, height - 1);
        v[k++] = input[(yy * width + xx) * 3 + c];
      }
    }
    // Sorting network (9 elements).
    sort2(&v[0], &v[1]); sort2(&v[3], &v[4]); sort2(&v[6], &v[7]);
    sort2(&v[1], &v[2]); sort2(&v[4], &v[5]); sort2(&v[7], &v[8]);
    sort2(&v[0], &v[1]); sort2(&v[3], &v[4]); sort2(&v[6], &v[7]);
    sort2(&v[0], &v[3]); sort2(&v[1], &v[4]); sort2(&v[2], &v[5]);
    sort2(&v[3], &v[6]); sort2(&v[4], &v[7]); sort2(&v[5], &v[8]);
    sort2(&v[0], &v[3]); sort2(&v[1], &v[4]); sort2(&v[2], &v[5]);
    sort2(&v[1], &v[3]); sort2(&v[2], &v[4]); sort2(&v[5], &v[7]);
    sort2(&v[2], &v[3]); sort2(&v[4], &v[5]);
    output[(y * width + x) * 3 + c] = v[4];
  }
}

__kernel void sobel(__global const uchar* input, __global uchar* output,
                    const int width, const int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;
  int xl = clamp(x - 1, 0, width - 1);
  int xr = clamp(x + 1, 0, width - 1);
  int yu = clamp(y - 1, 0, height - 1);
  int yd = clamp(y + 1, 0, height - 1);
  #define LUMA(xx, yy) ((int)((input[(yy * width + xx) * 3 + 0] * 30 + \
                               input[(yy * width + xx) * 3 + 1] * 59 + \
                               input[(yy * width + xx) * 3 + 2] * 11) / 100))
  int gx = -LUMA(xl, yu) - 2 * LUMA(xl, y) - LUMA(xl, yd)
         +  LUMA(xr, yu) + 2 * LUMA(xr, y) + LUMA(xr, yd);
  int gy = -LUMA(xl, yu) - 2 * LUMA(x, yu) - LUMA(xr, yu)
         +  LUMA(xl, yd) + 2 * LUMA(x, yd) + LUMA(xr, yd);
  int magnitude = abs(gx) + abs(gy);
  uchar v = (uchar)clamp(magnitude, 0, 255);
  output[(y * width + x) * 3 + 0] = v;
  output[(y * width + x) * 3 + 1] = v;
  output[(y * width + x) * 3 + 2] = v;
}
)CLC";

struct OpenClContext {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel invert_kernel = nullptr;
  cl_kernel median_kernel = nullptr;
  cl_kernel sobel_kernel = nullptr;
};

OpenClContext InitOpenCl() {
  OpenClContext ctx;
  cl_uint num_platforms = 0;
  clGetPlatformIDs(1, &ctx.platform, &num_platforms);
  if (num_platforms == 0) throw std::runtime_error("No OpenCL platform");
  cl_int err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device,
                              nullptr);
  if (err != CL_SUCCESS) {
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_CPU, 1, &ctx.device,
                        nullptr);
  }
  if (err != CL_SUCCESS) throw std::runtime_error("No OpenCL device");
  ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
  ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, &err);
  const char* src = kOpenClSource;
  ctx.program = clCreateProgramWithSource(ctx.context, 1, &src, nullptr, &err);
  err = clBuildProgram(ctx.program, 1, &ctx.device, "", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::size_t log_size = 0;
    clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, 0,
                          nullptr, &log_size);
    std::string log(log_size, '\0');
    clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG,
                          log_size, log.data(), nullptr);
    throw std::runtime_error("OpenCL build failed:\n" + log);
  }
  ctx.invert_kernel = clCreateKernel(ctx.program, "invert", &err);
  ctx.median_kernel = clCreateKernel(ctx.program, "median3x3", &err);
  ctx.sobel_kernel = clCreateKernel(ctx.program, "sobel", &err);
  return ctx;
}

void ReleaseOpenCl(OpenClContext& ctx) {
  clReleaseKernel(ctx.invert_kernel);
  clReleaseKernel(ctx.median_kernel);
  clReleaseKernel(ctx.sobel_kernel);
  clReleaseProgram(ctx.program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
}

ImageView RunOpenCl2d(const OpenClContext& ctx, cl_kernel kernel,
                      const ImageView& input) {
  const std::size_t bytes = input.pixels.size();
  cl_int err = 0;
  cl_mem in_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
  cl_mem out_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
  clEnqueueWriteBuffer(ctx.queue, in_buf, CL_TRUE, 0, bytes,
                       input.pixels.data(), 0, nullptr, nullptr);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
  clSetKernelArg(kernel, 2, sizeof(int), &input.width);
  clSetKernelArg(kernel, 3, sizeof(int), &input.height);
  std::size_t global[2] = {static_cast<std::size_t>(input.width),
                           static_cast<std::size_t>(input.height)};
  clEnqueueNDRangeKernel(ctx.queue, kernel, 2, nullptr, global, nullptr, 0,
                         nullptr, nullptr);
  Image output(bytes);
  clEnqueueReadBuffer(ctx.queue, out_buf, CL_TRUE, 0, bytes, output.data(), 0,
                      nullptr, nullptr);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);
  return {input.width, input.height, input.channels, std::move(output)};
}

ImageView InvertOpenCl(const OpenClContext& ctx, const ImageView& input) {
  const std::size_t bytes = input.pixels.size();
  cl_int err = 0;
  cl_mem in_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
  cl_mem out_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
  clEnqueueWriteBuffer(ctx.queue, in_buf, CL_TRUE, 0, bytes,
                       input.pixels.data(), 0, nullptr, nullptr);
  int total = static_cast<int>(bytes);
  clSetKernelArg(ctx.invert_kernel, 0, sizeof(cl_mem), &in_buf);
  clSetKernelArg(ctx.invert_kernel, 1, sizeof(cl_mem), &out_buf);
  clSetKernelArg(ctx.invert_kernel, 2, sizeof(int), &total);
  std::size_t global = static_cast<std::size_t>(bytes);
  clEnqueueNDRangeKernel(ctx.queue, ctx.invert_kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  Image output(bytes);
  clEnqueueReadBuffer(ctx.queue, out_buf, CL_TRUE, 0, bytes, output.data(), 0,
                      nullptr, nullptr);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);
  return {input.width, input.height, input.channels, std::move(output)};
}

template <typename Func>
double MeasureMilliseconds(Func&& func, int repeats) {
  using Clock = std::chrono::high_resolution_clock;
  func();
  double total_ms = 0.0;
  for (int i = 0; i < repeats; ++i) {
    const auto start = Clock::now();
    auto result = func();
    const auto end = Clock::now();
    total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    (void)result;
  }
  return total_ms / repeats;
}

struct BenchmarkRow {
  std::string filter;
  std::string backend;
  double avg_ms;
};

void PrintTable(const std::vector<BenchmarkRow>& rows) {
  std::cout << "\n" << std::left << std::setw(20) << "Filter"
            << std::setw(14) << "Backend" << std::right << std::setw(14)
            << "Avg ms" << "\n";
  std::cout << std::string(48, '-') << "\n";
  for (const auto& row : rows) {
    std::cout << std::left << std::setw(20) << row.filter << std::setw(14)
              << row.backend << std::right << std::setw(14) << std::fixed
              << std::setprecision(3) << row.avg_ms << "\n";
  }
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " input.png [repeats=5]\n";
      return 1;
    }
    const std::string input_path = argv[1];
    const int repeats = (argc >= 3) ? std::atoi(argv[2]) : 5;

    const ImageView input = LoadPng(input_path);
    std::cout << "Loaded " << input.width << "x" << input.height << " ("
              << input.pixels.size() / (1024.0 * 1024.0) << " MB)\n";

    namespace fs = std::filesystem;
    const fs::path input_stem = fs::path(input_path).stem();
    const fs::path out_dir = fs::path("output") / input_stem;
    fs::create_directories(out_dir);

    OpenClContext cl_ctx = InitOpenCl();

    auto bench = [&](const std::string& filter_name, const std::string& backend,
                     auto&& fn, const std::string& save_as = "") {
      const double avg = MeasureMilliseconds(fn, repeats);
      if (!save_as.empty()) SavePng((out_dir / save_as).string(), fn());
      return BenchmarkRow{filter_name, backend, avg};
    };

    std::vector<BenchmarkRow> rows;
    rows.push_back(bench("Inversion", "Sequential", [&] { return InvertSequential(input); }, "inversion.png"));
    rows.push_back(bench("Inversion", "OpenMP",     [&] { return InvertOpenMp(input); }));
    rows.push_back(bench("Inversion", "SIMD",       [&] { return InvertSimd(input); }));
    rows.push_back(bench("Inversion", "OpenCL",     [&] { return InvertOpenCl(cl_ctx, input); }));

    rows.push_back(bench("Median 3x3", "Sequential", [&] { return MedianSequential(input); }, "median.png"));
    rows.push_back(bench("Median 3x3", "OpenMP",     [&] { return MedianOpenMp(input); }));
    rows.push_back(bench("Median 3x3", "SIMD",       [&] { return MedianSimd(input); }));
    rows.push_back(bench("Median 3x3", "OpenCL",     [&] { return RunOpenCl2d(cl_ctx, cl_ctx.median_kernel, input); }));

    rows.push_back(bench("Sobel Edges", "Sequential", [&] { return EdgeSequential(input); }, "edges.png"));
    rows.push_back(bench("Sobel Edges", "OpenMP",     [&] { return EdgeOpenMp(input); }));
    rows.push_back(bench("Sobel Edges", "SIMD",       [&] { return EdgeSimd(input); }));
    rows.push_back(bench("Sobel Edges", "OpenCL",     [&] { return RunOpenCl2d(cl_ctx, cl_ctx.sobel_kernel, input); }));

    std::cout << "Saved results to " << out_dir.string() << "/\n";

    PrintTable(rows);
    std::FILE* csv = std::fopen("results.csv", "a");
    if (csv) {
      const double megapixels = (input.width * static_cast<double>(input.height)) / 1.0e6;
      for (const auto& r : rows) {
        std::fprintf(csv, "%s,%s,%d,%d,%.4f,%.4f\n", r.filter.c_str(),
                     r.backend.c_str(), input.width, input.height, megapixels,
                     r.avg_ms);
      }
      std::fclose(csv);
    }

    ReleaseOpenCl(cl_ctx);
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
