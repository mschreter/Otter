#pragma once
#include <deal.II/base/point.h>
#include <deal.II/base/mpi.h>

#include <tiffio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace Otter
{

template <int dim, typename Number = double>
std::pair<std::vector<Number>, std::array<int, dim>>
read_tiff_data_only(const std::string &filename)
{
  TIFF *tif = TIFFOpen(filename.c_str(), "r");
  if (!tif)
    throw std::runtime_error("Could not open TIFF file: " + filename);

  uint32_t width = 0, height = 0;
  uint16_t bitsPerSample = 0, samplesPerPixel = 0;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
  TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

  if (samplesPerPixel != 1)
    {
      TIFFClose(tif);
      throw std::runtime_error("Only samplesPerPixel == 1 supported.");
    }

  if (bitsPerSample != 1 && bitsPerSample != 8 && bitsPerSample != 16)
    {
      TIFFClose(tif);
      throw std::runtime_error("Only 1, 8, 16 bit TIFF supported.");
    }

  std::vector<Number> data;

  uint16_t n_slices = 1;

  if constexpr (dim == 3)
    n_slices = TIFFNumberOfDirectories(tif);

  const std::size_t total_size = static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) *
                                 static_cast<std::size_t>(n_slices);

  data.reserve(total_size);

  const tsize_t        lineSize = TIFFScanlineSize(tif);
  std::vector<uint8_t> buf(lineSize);

  for (uint16_t slice = 0; slice < n_slices; ++slice)
    {
      if constexpr (dim == 3)
        TIFFSetDirectory(tif, slice);

      for (uint32_t y = 0; y < height; ++y)
        {
          if (TIFFReadScanline(tif, buf.data(), y) < 0)
            throw std::runtime_error("Error reading scanline");

          if (bitsPerSample == 8)
            {
              const auto *p = reinterpret_cast<const uint8_t *>(buf.data());
              for (uint32_t x = 0; x < width; ++x)
                data.emplace_back(static_cast<Number>(p[x]));
            }
          else if (bitsPerSample == 16)
            {
              const auto *p = reinterpret_cast<const uint16_t *>(buf.data());
              for (uint32_t x = 0; x < width; ++x)
                data.emplace_back(static_cast<Number>(p[x]));
            }
          else if (bitsPerSample == 1)
            {
              const auto *row = reinterpret_cast<const uint8_t *>(buf.data());

              for (uint32_t x = 0; x < width; ++x)
                {
                  const uint32_t byte_i = x >> 3;
                  const uint32_t bit_i  = 7u - (x & 7u);
                  const uint8_t  bit    = (row[byte_i] >> bit_i) & 0x1u;

                  data.emplace_back(static_cast<Number>(bit));
                }
            }
        }
    }

  TIFFClose(tif);

  if constexpr (dim == 2)
    {
      return {data, std::array<int, dim>{{static_cast<int>(width), static_cast<int>(height)}}};
    }
  else
    {
      return {data,
              std::array<int, dim>{
                {static_cast<int>(width), static_cast<int>(height), static_cast<int>(n_slices)}}};
    }
}

/// Reads a grayscale TIFF and returns a vector of {x, y, (opt: z), value}.
template <int dim, typename Number = double>
std::pair<std::vector<dealii::Point<dim, Number>>, std::vector<std::vector<Number>>>
read_tiff(const std::string &filename,
          const Number       spacing_x = 1,
          const Number       spacing_y = 1,
          const Number       spacing_z = 1,
          const Number       x_shift   = 0,
          const Number       y_shift   = 0,
          const Number       z_shift   = 0)
{
  TIFF *tif = TIFFOpen(filename.c_str(), "r");
  if (!tif)
    throw std::runtime_error("Could not open TIFF file: " + filename);

  uint32_t width = 0, height = 0;
  uint16_t bitsPerSample = 0, samplesPerPixel = 0;
  uint16_t planarConfig = PLANARCONFIG_CONTIG;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  // sets bitsPerSample to default values (1) if it is not tagged (0)
  TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
  TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
  TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
  TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planarConfig);

  std::cout << "samples per pixel: " << samplesPerPixel << std::endl;
  std::cout << "bits per sample: " << bitsPerSample << std::endl;

  // We support:
  //  - 1 sample/pixel: grayscale
  //  - 2 samples/pixel: [z, value] interleaved
  if (samplesPerPixel != 1 && samplesPerPixel != 2)
    {
      TIFFClose(tif);
      throw std::runtime_error("Expected 1 or 2 samples per pixel (got " +
                               std::to_string(samplesPerPixel) + ")");
    }
  if (planarConfig != PLANARCONFIG_CONTIG)
    {
      TIFFClose(tif);
      throw std::runtime_error("Only PLANARCONFIG_CONTIG (interleaved samples) is supported.");
    }

  if (bitsPerSample != 1 && bitsPerSample != 8 && bitsPerSample != 16)
    {
      TIFFClose(tif);
      throw std::runtime_error("Only 1-bit, 8-bit or 16-bit TIFFs are supported.");
    }

  std::vector<dealii::Point<dim, Number>> points;
  std::vector<std::vector<Number>>        data;
  points.reserve(width * height);
  data.reserve(width * height);

  const tsize_t        lineSize = TIFFScanlineSize(tif);
  std::vector<uint8_t> buf(lineSize);

  if (bitsPerSample == 8)
    {
      if (samplesPerPixel == 1)
        {
          if constexpr (dim == 2)
            {
              // --- 2D grayscale ---
              for (uint32_t y = 0; y < height; ++y)
                {
                  if (TIFFReadScanline(tif, buf.data(), y) < 0)
                    throw std::runtime_error("Error reading scanline");

                  const auto *p8 = reinterpret_cast<const std::uint8_t *>(buf.data());

                  for (uint32_t x = 0; x < width; ++x)
                    {
                      const Number value = static_cast<Number>(p8[x]);

                      points.emplace_back(x_shift + x * spacing_x, y_shift + y * spacing_y);
                      data.push_back({value});
                    }
                }
            }
          else if constexpr (dim == 3)
            {
              // --- 3D grayscale volume (stack of directories) ---
              uint16_t n_slices = TIFFNumberOfDirectories(tif);

              const std::size_t n_voxels = static_cast<std::size_t>(width) *
                                           static_cast<std::size_t>(height) *
                                           static_cast<std::size_t>(n_slices);

              points.reserve(points.size() + n_voxels);
              data.reserve(data.size() + n_voxels);

              for (uint16_t slice = 0; slice < n_slices; ++slice)
                {
                  if (!TIFFSetDirectory(tif, slice))
                    throw std::runtime_error("TIFFSetDirectory failed");

                  const double z_coord = z_shift + static_cast<double>(slice) * spacing_z;

                  for (uint32_t y = 0; y < height; ++y)
                    {
                      if (TIFFReadScanline(tif, buf.data(), y) < 0)
                        throw std::runtime_error("Error reading scanline");

                      const auto *p8 = reinterpret_cast<const std::uint8_t *>(buf.data());

                      for (uint32_t x = 0; x < width; ++x)
                        {
                          const Number value = static_cast<Number>(p8[x]);

                          points.emplace_back(x_shift + x * spacing_x,
                                              y_shift + y * spacing_y,
                                              z_coord);
                          data.push_back({value});
                        }
                    }
                }
            }
          else
            {
              AssertThrow(false, dealii::ExcNotImplemented());
            }
        }
      else if (samplesPerPixel == 2)
        {
          // --- interleaved [z,value] ---
          if constexpr (dim == 3)
            {
              for (uint32_t y = 0; y < height; ++y)
                {
                  if (TIFFReadScanline(tif, buf.data(), y) < 0)
                    throw std::runtime_error("Error reading scanline");

                  const auto *p8 = reinterpret_cast<const std::uint8_t *>(buf.data());

                  for (uint32_t x = 0; x < width; ++x)
                    {
                      const std::uint8_t z_raw    = p8[2 * x + 0];
                      const std::uint8_t gray_raw = p8[2 * x + 1];

                      const double z     = static_cast<double>(z_raw);
                      const Number value = static_cast<Number>(gray_raw);

                      points.emplace_back(x_shift + x * spacing_x,
                                          y_shift + y * spacing_y,
                                          z_shift + z * spacing_z);

                      data.push_back({value});
                    }
                }
            }
          else
            {
              AssertThrow(false, dealii::ExcNotImplemented());
            }
        }
      else
        {
          TIFFClose(tif);
          throw std::runtime_error("Unsupported samplesPerPixel for 8-bit TIFF.");
        }
    }
  else if (bitsPerSample == 16)
    {
      if (samplesPerPixel == 1)
        {
          if constexpr (dim == 2)
            {
              // --- 2D grayscale (your existing behavior, single slice) ---
              for (uint32_t y = 0; y < height; ++y)
                {
                  if (TIFFReadScanline(tif, buf.data(), y) < 0)
                    throw std::runtime_error("Error reading scanline");

                  auto *p16 = reinterpret_cast<std::uint16_t *>(buf.data());

                  for (uint32_t x = 0; x < width; ++x)
                    {
                      const std::uint16_t v     = p16[x];
                      const Number        value = static_cast<Number>(v);

                      points.emplace_back(x_shift + x * spacing_x, y_shift + y * spacing_y);
                      data.push_back({value});
                    }
                }
            }
          else if constexpr (dim == 3)
            {
              // --- 3D grayscale volume: stack of slices ---
              // We assume width/height identical for all slices.
              uint16_t n_slices = TIFFNumberOfDirectories(tif);

              const std::size_t n_voxels = static_cast<std::size_t>(width) *
                                           static_cast<std::size_t>(height) *
                                           static_cast<std::size_t>(n_slices);

              points.reserve(points.size() + n_voxels);
              data.reserve(data.size() + n_voxels);

              for (uint16_t slice = 0; slice < n_slices; ++slice)
                {
                  if (!TIFFSetDirectory(tif, slice))
                    throw std::runtime_error("TIFFSetDirectory failed");

                  const Number z_coord = z_shift + static_cast<Number>(slice) * spacing_z;

                  for (uint32_t y = 0; y < height; ++y)
                    {
                      if (TIFFReadScanline(tif, buf.data(), y) < 0)
                        throw std::runtime_error("Error reading scanline");

                      auto *p16 = reinterpret_cast<std::uint16_t *>(buf.data());

                      for (uint32_t x = 0; x < width; ++x)
                        {
                          const std::uint16_t v     = p16[x];
                          const Number        value = static_cast<Number>(v);

                          points.emplace_back(x_shift + x * spacing_x,
                                              y_shift + y * spacing_y,
                                              z_coord);
                          data.push_back({value});
                        }
                    }
                }
            }
          else
            {
              AssertThrow(false, dealii::ExcNotImplemented());
            }
        }
      else if (samplesPerPixel == 2)
        {
          // --- your old "z,value interleaved" mode, keep as is ---
          auto *p16 = reinterpret_cast<std::uint16_t *>(buf.data());

          for (uint32_t y = 0; y < height; ++y)
            {
              if (TIFFReadScanline(tif, buf.data(), y) < 0)
                throw std::runtime_error("Error reading scanline");

              for (uint32_t x = 0; x < width; ++x)
                {
                  const std::uint16_t z_raw    = p16[2 * x + 0];
                  const std::uint16_t gray_raw = p16[2 * x + 1];

                  const Number z     = static_cast<Number>(z_raw);
                  const Number value = static_cast<Number>(gray_raw);

                  if constexpr (dim == 3)
                    {
                      points.emplace_back(x_shift + x * spacing_x,
                                          y_shift + y * spacing_y,
                                          z_shift + z * spacing_z);
                      data.push_back({value});
                    }
                  else
                    {
                      AssertThrow(false, dealii::ExcNotImplemented());
                    }
                }
            }
        }
    }
  else if (bitsPerSample == 1)
    {
      // 1-bit bilevel: pixels packed MSB->LSB in each byte.
      auto get_bit = [&](const std::uint8_t *row, std::uint32_t x) -> std::uint8_t {
        const std::uint32_t byte_i = x >> 3;        // x / 8
        const std::uint32_t bit_i  = 7u - (x & 7u); // MSB first
        return (row[byte_i] >> bit_i) & 0x1u;
      };

      if (samplesPerPixel != 1)
        {
          TIFFClose(tif);
          throw std::runtime_error("bitsPerSample=1 is only supported for samplesPerPixel=1.");
        }

      if constexpr (dim == 2)
        {
          for (uint32_t y = 0; y < height; ++y)
            {
              if (TIFFReadScanline(tif, buf.data(), y) < 0)
                throw std::runtime_error("Error reading scanline");

              const auto *row = reinterpret_cast<const std::uint8_t *>(buf.data());

              for (uint32_t x = 0; x < width; ++x)
                {
                  const Number value = static_cast<Number>(get_bit(row, x)); // 0 or 1

                  points.emplace_back(x_shift + x * spacing_x, y_shift + y * spacing_y);
                  data.push_back({value});
                }
            }
        }
      else if constexpr (dim == 3)
        {
          const uint16_t n_slices = TIFFNumberOfDirectories(tif);

          const std::size_t n_voxels = static_cast<std::size_t>(width) *
                                       static_cast<std::size_t>(height) *
                                       static_cast<std::size_t>(n_slices);

          points.reserve(points.size() + n_voxels);
          data.reserve(data.size() + n_voxels);

          for (uint16_t slice = 0; slice < n_slices; ++slice)
            {
              if (!TIFFSetDirectory(tif, slice))
                throw std::runtime_error("TIFFSetDirectory failed");

              const double z_coord = z_shift + static_cast<double>(slice) * spacing_z;

              for (uint32_t y = 0; y < height; ++y)
                {
                  if (TIFFReadScanline(tif, buf.data(), y) < 0)
                    throw std::runtime_error("Error reading scanline");

                  const auto *row = reinterpret_cast<const std::uint8_t *>(buf.data());

                  for (uint32_t x = 0; x < width; ++x)
                    {
                      const Number value = static_cast<Number>(get_bit(row, x)); // 0 or 1

                      points.emplace_back(x_shift + x * spacing_x,
                                          y_shift + y * spacing_y,
                                          z_coord);
                      data.push_back({value});
                    }
                }
            }
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
        }
    }
  else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

  TIFFClose(tif);

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "tiff file read" << std::endl;

  return {points, data};
}

inline int
clamp_int(const int v, const int lo, const int hi)
{
  return std::max(lo, std::min(v, hi));
}

// Half-up rounding: floor(x + 0.5) works for x>=0.
// This version works symmetrically for negative values too.
inline int
round_to_int_symmetric(const double x)
{
  return (x >= 0.0) ? static_cast<int>(std::floor(x + 0.5)) : static_cast<int>(std::ceil(x - 0.5));
}

template <int dim>
struct ClosestGridPoint
{
  static_assert(dim == 2 || dim == 3, "Only dim=2 or dim=3 supported.");

  std::array<int, dim> indices;          // nearest grid indices
  std::size_t          linear_index = 0; // row-major linear index (x fastest)
  dealii::Point<dim>   point;            // coordinate of closest grid point
};

/**
 * Closest regular-grid point to query in dim=2/3.
 *
 * Grid definition:
 *  - origin: coordinate of index (0,0[,0])
 *  - spacing: hx,hy[,hz]
 *  - n: number of points along each axis: nx,ny[,nz]
 *
 * Ordering / linearization:
 *  - x fastest, then y, then z (row-major):
 *    id = ix + nx*(iy + ny*iz) in 3D
 *    id = ix + nx*iy           in 2D
 */
template <int dim>
ClosestGridPoint<dim>
closest_point_regular_grid(const dealii::Point<dim>   &query,
                           const std::array<int, dim> &n_points,
                           const dealii::Point<dim>   &origin = dealii::Point<dim>())
{
  static_assert(dim == 2 || dim == 3, "Only dim=2 or dim=3 supported.");

  dealii::Point<dim> spacing;

  for (int d = 0; d < dim; ++d)
    spacing[d] = 1;

  ClosestGridPoint<dim> out;

  // Compute nearest integer indices with clamping
  for (int d = 0; d < dim; ++d)
    {
      const double fd = (query[d] - origin[d]) / spacing[d];
      int          id = round_to_int_symmetric(fd);
      id              = clamp_int(id, 0, n_points[d] - 1);
      out.indices[d]  = id;
    }

  // Compute closest grid point coordinates
  for (int d = 0; d < dim; ++d)
    out.point[d] = origin[d] + static_cast<double>(out.indices[d]) * spacing[d];

  // Row-major linear index (x fastest)
  // id = i0 + n0*(i1 + n1*i2) etc.
  std::size_t linear = static_cast<std::size_t>(out.indices[0]);
  std::size_t stride = static_cast<std::size_t>(n_points[0]);

  for (int d = 1; d < dim; ++d)
    {
      linear += stride * static_cast<std::size_t>(out.indices[d]);
      stride *= static_cast<std::size_t>(n_points[d]);
    }

  out.linear_index = linear;
  return out;
}
}
