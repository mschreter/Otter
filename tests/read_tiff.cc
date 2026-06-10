#include <deal.II/base/point.h>

#include <tiffio.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dealii;

/// Reads a grayscale TIFF and returns a vector of {x, y, (opt: z), value}.
template <int dim>
std::vector<std::pair<dealii::Point<dim>, double>>
read_tiff(const std::string &filename,
          const double       spacing_x = 1,
          const double       spacing_y = 1,
          const double       spacing_z = 1)
{
  TIFF *tif = TIFFOpen(filename.c_str(), "r");
  if (!tif)
    throw std::runtime_error("Could not open TIFF file: " + filename);

  uint32_t width = 0, height = 0;
  uint16_t bitsPerSample = 0, samplesPerPixel = 0;
  uint16_t planarConfig = PLANARCONFIG_CONTIG;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
  TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
  TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planarConfig);

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

  if (bitsPerSample != 8 && bitsPerSample != 16)
    {
      TIFFClose(tif);
      throw std::runtime_error("Only 8-bit or 16-bit TIFFs are supported.");
    }

  std::vector<std::pair<dealii::Point<dim>, double>> data;
  data.reserve(width * height);

  const tsize_t        lineSize = TIFFScanlineSize(tif);
  std::vector<uint8_t> buf(lineSize);

  for (uint32_t y = 0; y < height; ++y)
    {
      if (TIFFReadScanline(tif, buf.data(), y) < 0)
        throw std::runtime_error("Error reading scanline");

      if (bitsPerSample == 8)
        {
          // buffer layout for samplesPerPixel=1: [v0, v1, v2, ...]
          // buffer layout for samplesPerPixel=2: [z0, g0, z1, g1, ...]
          const std::uint8_t *p = buf.data();

          for (uint32_t x = 0; x < width; ++x)
            {
              double z     = 0.0;
              double value = 0.0;

              if (samplesPerPixel == 1)
                {
                  value = static_cast<double>(p[x]);
                  z     = value;

                  if constexpr (dim == 2)
                    data.push_back({Point<dim>(x * spacing_x, y * spacing_y), value});
                  else
                    AssertThrow(false, dealii::ExcNotImplemented());
                }
              else // samplesPerPixel == 2
                {
                  const std::uint8_t z_raw    = p[2 * x + 0];
                  const std::uint8_t gray_raw = p[2 * x + 1];

                  z     = static_cast<double>(z_raw);
                  value = static_cast<double>(gray_raw);

                  if constexpr (dim == 3)
                    data.push_back(
                      {Point<dim>(x * spacing_x, y * spacing_y, z * spacing_z), value});
                  else
                    AssertThrow(false, dealii::ExcNotImplemented());
                }
            }
        }
      else if (bitsPerSample == 16)
        {
          // buffer is an array of uint16_t
          auto *p16 = reinterpret_cast<std::uint16_t *>(buf.data());

          for (uint32_t x = 0; x < width; ++x)
            {
              double value = 0.0;

              if (samplesPerPixel == 1)
                {
                  const std::uint16_t v = p16[x];
                  value                 = static_cast<double>(v);
                  if constexpr (dim == 2)
                    data.push_back({Point<dim>(x * spacing_x, y * spacing_y), value});
                  else
                    AssertThrow(false, dealii::ExcNotImplemented());
                }
              else // samplesPerPixel == 2
                {
                  const std::uint16_t z_raw    = p16[2 * x + 0];
                  const std::uint16_t gray_raw = p16[2 * x + 1];

                  double z = static_cast<double>(z_raw);
                  value    = static_cast<double>(gray_raw);

                  if constexpr (dim == 3)
                    data.push_back(
                      {Point<dim>(x * spacing_x, y * spacing_y, z * spacing_z), value});
                  else
                    AssertThrow(false, dealii::ExcNotImplemented());
                }
            }
        }
    }

  TIFFClose(tif);
  return data;
}
#include <iostream>

int
main()
{
  auto xyzv = read_tiff<2>(SOURCE_DIR "/example.tif", 0.352778, 0.352778, 0.352778);

  std::cout << "Read " << xyzv.size() << " points\n";
  if (!xyzv.empty())
    {
      {
        auto p   = xyzv.front().first;
        auto val = xyzv.front().second;
        std::cout << "First point: x=";

        for (int d = 0; d < 2; d++)
          std::cout << p[d] << " ";

        std::cout << " value=" << val << "\n";
      }
      {
        auto p   = xyzv.back().first;
        auto val = xyzv.back().second;
        std::cout << "Last point: x=";

        for (int d = 0; d < 2; d++)
          std::cout << p[d] << " ";

        std::cout << " value=" << val << "\n";
      }
    }
}
