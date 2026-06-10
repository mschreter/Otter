#pragma once
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>


template <int dim, typename Number>
class SphericalParticalPacking : public dealii::Function<dim, Number>
{
public:
  SphericalParticalPacking(const std::vector<dealii::Point<dim, Number>> &centers,
                           const std::vector<Number>                      radius,
                           const Number                                   rhs_value = Number(1.0))
    : dealii::Function<dim, Number>()
    , centers(centers)
    , radii(radius)
    , rhs_value(rhs_value)
  {}

  Number
  value(const dealii::Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    bool point_is_in_sphere = false;

    for (unsigned int i = 0; i < centers.size(); ++i)
      {
        const Number distance = (p - centers[i]).norm();

        if (distance <= radii[i])
          {
            point_is_in_sphere = true;
            break;
          }
      }
    return point_is_in_sphere ? rhs_value : 0;
  }

private:
  const std::vector<dealii::Point<dim, Number>> centers;
  const std::vector<Number>                     radii;
  const Number                                  rhs_value;
};

template <int dim>
std::pair<std::vector<dealii::Point<dim>>, std::vector<double>>
read_spherical_packing_data(const std::string &filename)
{
  std::ifstream file(filename);
  AssertThrow(file, dealii::ExcMessage("Could not open file: " + filename));

  std::vector<dealii::Point<dim>> points;
  std::vector<double>             values;

  std::string line;

  while (std::getline(file, line))
    {
      if (line.empty())
        continue;

      // replace commas with spaces (supports CSV and TXT)
      for (char &c : line)
        if (c == ',')
          c = ' ';

      std::stringstream ss(line);

      double coords[dim];
      double value;

      for (unsigned int d = 0; d < dim; ++d)
        {
          ss >> coords[d];
          Assert(ss, dealii::ExcMessage("Not enough coordinate columns in line: " + line));
        }

      ss >> value;
      Assert(ss, dealii::ExcMessage("Missing value column in line: " + line));


      dealii::Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = coords[d];

      points.push_back(p);
      values.push_back(value);
    }

  return {points, values};
}

// Pretty printer with MPI aggregates for VmRSS
DEAL_II_ALWAYS_INLINE inline void
print_memory_stats(const dealii::Utilities::System::MemoryStats &before,
                   const dealii::Utilities::System::MemoryStats &after,
                   MPI_Comm                                      comm = MPI_COMM_WORLD,
                   const std::string                             id   = "")
{
  const int rank   = dealii::Utilities::MPI::this_mpi_process(comm);
  const int nprocs = dealii::Utilities::MPI::n_mpi_processes(comm);

  // local values (MB)
  const double before_mb = before.VmRSS / 1024.0 / 1024;
  const double after_mb  = after.VmRSS / 1024.0 / 1024;
  const double delta_mb  = (after.VmRSS - before.VmRSS) / 1024.0 / 1024;

  // --- per-rank line (optional: guard with rank==0 if too chatty) ---
  std::cout << std::fixed << std::setprecision(2);

  // --- global aggregates ---
  double sum_before_mb = 0, max_before_mb = 0, sum_after_mb = 0.0, max_after_mb = 0.0,
         sum_delta_mb = 0.0;
  MPI_Allreduce(&before_mb, &sum_before_mb, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&before_mb, &max_before_mb, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&after_mb, &sum_after_mb, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&after_mb, &max_after_mb, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&delta_mb, &sum_delta_mb, 1, MPI_DOUBLE, MPI_SUM, comm);

  if (rank == 0)
    {
      std::cout << " [" << id << "]  [aggregate] VmRSS before: "
                << " [" << id << "]  SUM=" << sum_before_mb << " GB, "
                << " [" << id << "]  AVG=" << (sum_before_mb / nprocs) << " GB/rank, "
                << " [" << id << "]  MAX=" << max_before_mb << " GB\n";
      std::cout << " [" << id << "]  [aggregate] VmRSS after: "
                << " [" << id << "]  SUM=" << sum_after_mb << " GB, "
                << " [" << id << "]  AVG=" << (sum_after_mb / nprocs) << " GB/rank, "
                << " [" << id << "]  MAX=" << max_after_mb << " GB\n";
      std::cout << " [" << id << "]  [aggregate] Δ VmRSS: "
                << " [" << id << "]  SUM=" << sum_delta_mb << " GB, "
                << " [" << id << "]  AVG=" << (sum_delta_mb / nprocs) << " GB/rank\n";
    }
}

template <int dim, typename Number>
void
write_vtk_with_points(const std::vector<dealii::Point<dim, Number>> &points,
                      const std::vector<Number>                     &values,
                      const std::string                             &filename)
{
  std::ofstream out(filename);
  out << "# vtk DataFile Version 3.0\n";
  out << "VTK output\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n";
  out << "POINTS " << points.size() << " float\n";
  for (const auto &p : points)
    {
      for (unsigned int d = 0; d < dim; ++d)
        out << p[d] << " ";

      out << "\n";
    }

  out << "\nVERTICES " << points.size() << " " << 2 * points.size() << "\n";
  for (size_t i = 0; i < points.size(); ++i)
    out << "1 " << i << "\n";

  out << "\nPOINT_DATA " << values.size() << "\n";
  out << "SCALARS value float 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto &v : values)
    out << v << "\n";
}

// TODO: the 0th component is hardcoded
template <int dim, typename Number>
void
write_vtk_with_points(const std::vector<dealii::Point<dim, Number>> &points,
                      const std::vector<std::vector<Number>>        &values,
                      const std::string                             &filename)
{
  std::ofstream out(filename);
  out << "# vtk DataFile Version 3.0\n";
  out << "VTK output\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n";
  out << "POINTS " << points.size() << " float\n";
  for (const auto &p : points)
    {
      for (unsigned int d = 0; d < dim; ++d)
        out << p[d] << " ";

      out << "\n";
    }

  out << "\nVERTICES " << points.size() << " " << 2 * points.size() << "\n";
  for (size_t i = 0; i < points.size(); ++i)
    out << "1 " << i << "\n";

  out << "\nPOINT_DATA " << values.size() << "\n";
  out << "SCALARS value float 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto &v : values)
    out << v[0] << "\n";
}
