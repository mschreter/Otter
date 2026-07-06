#include <deal.II/base/point.h>

#include <otter/read_tiff.h>
#include <tiffio.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dealii;

int
main()
{
  const auto xyzv =
    Otter::read_tiff<2>(SOURCE_DIR "/points/example.tif", 0.352778, 0.352778, 0.352778);

  const auto &points = xyzv.first;
  const auto &values = xyzv.second;

  std::cout << "Read " << points.size() << " points\n";

  AssertThrow(points.size() == values.size(),
              ExcMessage("Number of TIFF points and values does not match."));

  if (!points.empty())
    {
      {
        const auto &p   = points.front();
        const auto &val = values.front();

        std::cout << "First point: x=";

        for (unsigned int d = 0; d < 2; ++d)
          std::cout << p[d] << " ";

        std::cout << " value=" << val[0] << "\n";
      }

      {
        const auto &p   = points.back();
        const auto &val = values.back();

        std::cout << "Last point: x=";

        for (unsigned int d = 0; d < 2; ++d)
          std::cout << p[d] << " ";

        std::cout << " value=" << val[0] << "\n";
      }
    }
}
