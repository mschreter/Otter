#pragma once
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_handler.h>
#ifdef DEAL_II_WITH_ARBORX
#  include <deal.II/arborx/distributed_tree.h>
#endif
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/rtree.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <mpi.h>
#include <read_tiff.h>
#include <utils.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>


// Make global particle properties vector on process 0
namespace Otter
{

template <int dim, typename Number>
inline std::pair<std::vector<dealii::Point<dim, Number>>, std::vector<std::vector<Number>>>
get_particles_and_properties(std::string    particle_data_file,
                             unsigned int   n_properties,
                             char           delimiter        = ' ',
                             const MPI_Comm mpi_communicator = MPI_COMM_WORLD)
{
  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
  // Make global particle properties vector
  std::vector<dealii::Point<dim, Number>> particle_locations;
  std::vector<std::vector<Number>>        properties{};

  // serial read of xyz
  // TODO: make parallel
  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      // TODO: create read_xyz function
      if (particle_data_file.ends_with(".xyz"))
        {
          std::fstream file;
          file.open(particle_data_file, std::ios::in);
          AssertThrow(!(file.fail()),
                      dealii::ExcMessage("Unable to open particle data file \"" +
                                         particle_data_file + "\". Aborting!"));
          std::string line;
          std::getline(file, line); // Ignore the first line
          while (std::getline(file, line))
            {
              std::vector<Number> particle_properties(n_properties, 0.0);
              std::string         temp;
              std::istringstream  data_string(line);
              // particle position
              dealii::Point<dim, Number> location;
              for (unsigned int i = 0; i < dim; i++)
                {
                  std::getline(data_string, temp, delimiter);
                  location[i] = std::stod(temp);
                }
              particle_locations.push_back(location);
              for (unsigned int i = 0; i < n_properties; i++)
                {
                  std::getline(data_string, temp, delimiter);
                  particle_properties[i] = std::stod(temp);
                }
              properties.push_back(particle_properties);
            }
        }
      else if (particle_data_file.ends_with(".tif"))
        {
          // TODO: add parameters
          auto data = read_tiff<dim, Number>(particle_data_file, 1, 1, 1, 0, 0, 0);
          // auto data = read_tiff<dim, Number>(particle_data_file, 0.33, -0.33, -1, 0, 330, 420);
          particle_locations.swap(data.first);
          properties.swap(data.second);
        }
      else
        AssertThrow(false, dealii::ExcMessage("Only *.xyz or *.tif files supported."));

print_entry(pcout, "Properties read", properties.size());
    }

  return {particle_locations, properties};
}

/**
 * @brief Initialize the particle handler from an input file.
 *
 * This function reads particle data from a specified input file and initializes
 * the given `particle_handler` with the provided particle positions and
 * associated property values. The particles are inserted into the domain
 * defined by the provided triangulation and mapping.
 *
 * The Number of properties per particle is specified explicitly, and the
 * function will read that many values for each particle from the file. Each
 * line in the input file should contain `dim` position coordinates followed by
 * `n_properties` property values, separated by the specified delimiter.
 *
 * @tparam dim The spatial dimension of the problem.
 * @tparam Number The numeric type used for particle coordinates and properties.
 *
 * @param[out] particle_handler The particle handler to be populated with
 * particle data.
 * @param[in] triangulation The triangulation defining the spatial domain.
 * @param[in] mapping The mapping used for placing particles in the curved
 * domain.
 * @param[in] particle_data_file Path to the input file containing particle
 * data.
 * @param[in] n_properties The Number of properties associated with each
 * particle.
 * @param[in] delimiter The character used to separate fields in the data file
 * (default is space).
 * @param[in] mpi_communicator The MPI communicator used for parallel
 * distribution (default is MPI_COMM_WORLD).
 */
template <int dim, typename Number>
inline void
initialize_particle_handler(dealii::Particles::ParticleHandler<dim> &particle_handler,
                            const dealii::Triangulation<dim>        &triangulation,
                            const dealii::Mapping<dim>              &mapping,
                            std::string                              particle_data_file,
                            unsigned int                             n_properties,
                            char                                     delimiter = ' ',
                            const MPI_Comm mpi_communicator                    = MPI_COMM_WORLD)
{
  const auto data =
    get_particles_and_properties<dim, Number>(particle_data_file, n_properties, delimiter);

  const auto &particle_locations = data.first;
  const auto &properties         = data.second;

  // Make the particles from the data in @struct particle_data
  std::vector<dealii::BoundingBox<dim>> local_bounding_box =
    dealii::GridTools::compute_mesh_predicate_bounding_box(
      triangulation, dealii::IteratorFilters::LocallyOwnedCell());
  std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_box =
    dealii::Utilities::MPI::all_gather(mpi_communicator, local_bounding_box);

  particle_handler.initialize(triangulation, mapping, n_properties);

  particle_handler.insert_global_particles(dealii::Utilities::MPI::this_mpi_process(
                                             mpi_communicator) == 0 ?
                                             particle_locations :
                                             std::vector<dealii::Point<dim, Number>>{},
                                           global_bounding_box,
                                           properties);
  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "Particles global insertion finished: " << particle_handler.n_global_particles()
              << std::endl;
}

/**
 * @brief Assemble a right-hand side vector from particle data using matrix-free
 * evaluation.
 *
 * Initializes the given `particle_handler` from a file and distributes scalar
 * particle values onto the finite element space using point evaluations. The
 * result is stored in `rhs`, using a matrix-free loop and simple mass-weighted
 * integration over cells.
 *
 * @tparam dim Spatial dimension.
 * @tparam Number Scalar type.
 * @tparam VectorType Type of the output vector.
 *
 * @param[out] rhs Right-hand side vector to fill.
 * @param[in,out] particle_handler Particle handler to initialize and use.
 * @param[in] mapping Mapping from reference to physical space.
 * @param[in] triangulation The mesh over which particles are defined.
 * @param[in] particle_file Path to the particle input file.
 * @param[in] matrix_free Matrix-free data for integration.
 * @param[in] zero_out Whether to zero the output vector before assembly
 * (default: true).
 */
template <int dim, typename Number, typename VectorType>
inline void
create_rhs_from_solid_particles(
  VectorType                                                             &rhs,
  const dealii::Mapping<dim>                                             &mapping,
  const dealii::Triangulation<dim>                                       &triangulation,
  std::string                                                             particle_file,
  const dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> &matrix_free,
  std::string                                                             particle_output_file,
  const bool                                                              enable_output,
  const bool                                                              enable_memory_stats,
  const bool                                                              zero_out = true)
{
  using namespace dealii;

  Utilities::System::MemoryStats mem_before, mem_after;
  dealii::Utilities::System::get_memory_stats(mem_before);

  using VectorizedArrayType = VectorizedArray<Number>;
  dealii::Particles::ParticleHandler<dim> particle_handler;

  initialize_particle_handler<dim, Number>(
    particle_handler, triangulation, mapping, particle_file, 1 /*n_components*/, ' ' /*delimiter*/);

  VectorType dummy;

  matrix_free.template cell_loop<VectorType, VectorType>(
    [&](const auto &data, auto &dst, const auto & /*src*/, const auto cell_range) {
      FEPointEvaluation<1, dim, dim, Number> fe_point_eval(mapping,
                                                           data.get_dof_handler().get_fe(),
                                                           update_values);

      std::vector<Number> local_values(data.get_dof_handler().get_fe().n_dofs_per_cell());
      std::vector<types::global_dof_index> local_dof_indices(
        data.get_dof_handler().get_fe().n_dofs_per_cell());

      for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second; ++cell_batch)
        {
          for (unsigned int lane = 0; lane < data.n_active_entries_per_cell_batch(cell_batch);
               ++lane)
            {
              const auto cell_volume = data.get_cell_iterator(cell_batch, lane)->measure();
              data.get_cell_iterator(cell_batch, lane)
                ->as_dof_handler_iterator(data.get_dof_handler())
                ->get_dof_indices(local_dof_indices);

              auto n_points =
                particle_handler.n_particles_in_cell(data.get_cell_iterator(cell_batch, lane));

              if (n_points == 0)
                continue;

              std::vector<Point<dim, Number>> unit_points;
              std::vector<Number>             val;
              for (const auto &p :
                   particle_handler.particles_in_cell(data.get_cell_iterator(cell_batch, lane)))
                {
                  unit_points.emplace_back(p.get_reference_location());
                  val.emplace_back(p.get_properties()[0]);
                }

              fe_point_eval.reinit(data.get_cell_iterator(cell_batch, lane), unit_points);

              for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                fe_point_eval.submit_value(val[q] * cell_volume / n_points, q);

              fe_point_eval.test_and_sum(local_values, EvaluationFlags::values);
              AffineConstraints<Number>().distribute_local_to_global(local_values,
                                                                     local_dof_indices,
                                                                     dst);
            }
        }
    },
    rhs,
    dummy,
    zero_out);

  // write output
  if (enable_output)
    {
      std::vector<std::string> solution_names(1, "value");
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        particle_data_component_interpretation(
          1, dealii::DataComponentInterpretation::component_is_scalar);
      dealii::Particles::DataOut<dim, dim> particle_output;
      particle_output.build_patches(particle_handler,
                                    solution_names,
                                    particle_data_component_interpretation);

      particle_output.write_vtu_in_parallel(particle_output_file, MPI_COMM_WORLD);
    }

  if (enable_memory_stats)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "create_rhs");
    }
}

template <int dim, typename Number, typename NumberCPP, typename VectorType>
inline void
create_rhs_from_solid_particles_closest_point(
  VectorType &rhs,
  const dealii::Mapping<dim> & /*mapping*/,
  std::string                                                             particle_file,
  const dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> &matrix_free,
  std::string                                                             particle_output_file,
  const bool                                                              enable_output,
  const bool                                                              enable_memory_stats,
  const bool                                                              zero_out = true)
{
  using namespace dealii;
  Utilities::System::MemoryStats mem_before, mem_after, mem_after_ptree;
  dealii::Utilities::System::get_memory_stats(mem_before);

  // read the particle data on process 0
  const auto data =
    get_particles_and_properties<dim, NumberCPP>(particle_file,
                                                 1 /*n_properties*/); //, " " /*delimiter*/);

  auto particle_locations = data.first;
  auto properties         = data.second;

  if (enable_output)
    write_vtk_with_points<dim, NumberCPP>(particle_locations, properties, particle_output_file);

  // distribute properties on all processes
  particle_locations = Utilities::MPI::broadcast(MPI_COMM_WORLD, particle_locations, 0);

  properties = Utilities::MPI::broadcast(MPI_COMM_WORLD, properties, 0);

  using ContainerType = std::vector<BoundingBox<dim, NumberCPP>>;

  ContainerType properties_tree;
  for (const auto &p : particle_locations)
    properties_tree.emplace_back(BoundingBox<dim, NumberCPP>(p));

  const auto tree_particles = pack_rtree_of_indices(properties_tree);

  if (enable_memory_stats)
    {
      dealii::Utilities::System::get_memory_stats(mem_after_ptree);
      print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "after ptree");
    }

  using VectorizedArrayType = VectorizedArray<Number>;
  using FECellIntegrator    = FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  VectorType dummy;

  matrix_free.template cell_loop<VectorType, VectorType>(
    [&](const auto &data, auto &dst, const auto & /*src*/, const auto cell_range) {
      FECellIntegrator phi(data);

      for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second; ++cell_batch)
        {
          phi.reinit(cell_batch);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto point_batch = phi.quadrature_point(q);

              // perform closest point search
              VectorizedArray<Number> val;
              for (unsigned int lane = 0; lane < data.n_active_entries_per_cell_batch(cell_batch);
                   ++lane)
                {
                  Point<dim, Number> quad_point;
                  for (unsigned int d = 0; d < dim; ++d)
                    quad_point[d] = point_batch[d][lane];

                  std::vector<typename ContainerType::size_type> closest_particle;
                  tree_particles.query(boost::geometry::index::nearest(quad_point, 1),
                                       std::back_inserter(closest_particle));

                  if (closest_particle.size() != 1)
                    {
                      std::cout << "WARNING: The Number of found nearest points is "
                                   "wrong. We found"
                                << closest_particle.size() << " points. The quadrature point is ";
                      for (unsigned int d = 0; d < dim; ++d)
                        std::cout << quad_point[d] << " ";

                      std::cout << std::endl;
                    }

                  val[lane] = properties[closest_particle[0]][0];
                }

              phi.submit_value(val, q);
            }

          phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    },
    rhs,
    dummy,
    zero_out);

  if (enable_memory_stats)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "create_rhs");
    }
}

template <int dim, typename Number, typename VectorType>
inline void
create_rhs_from_solid_particles_closest_point_fast(
  VectorType                                                             &rhs,
  const dealii::Mapping<dim>                                             &mapping,
  std::string                                                             particle_file,
  const dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> &matrix_free,
  std::string                                                             particle_output_file,
  const bool                                                              is_regular_grid,
  const bool                                                              enable_output,
  const bool                                                              enable_memory_stats,
  const bool                                                              zero_out = true)
{
  using namespace dealii;
  Utilities::System::MemoryStats mem_before, mem_after;
  dealii::Utilities::System::get_memory_stats(mem_before);

  std::vector<double> properties;
  if (is_regular_grid)
    {
      AssertThrow(particle_file.ends_with(".tif"),
                  ExcMessage("this option is only supported for tiff files."));
      // 1) read properties from tiff file
      std::array<int, dim> n_points_tiff;
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "MEMORY EFFICIENT PATH: store only data from tiff file" << std::endl;
          const auto &data = read_tiff_data_only<dim>(particle_file);
          n_points_tiff    = data.second;
          properties       = data.first;
          // communicate the number of points to all processes
        }

      n_points_tiff = Utilities::MPI::broadcast(MPI_COMM_WORLD, n_points_tiff, 0);

      // 2) collect quadrature points and perform closest point search
      std::vector<std::size_t> linear_index; // row-major linear index (x fastest)

      using VectorizedArrayType = VectorizedArray<Number>;
      using FECellIntegrator    = FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

      VectorType dummy;

      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &data, auto &dst, const auto & /*src*/, const auto cell_range) {
          FECellIntegrator phi(data);

          for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
               ++cell_batch)
            {
              phi.reinit(cell_batch);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  const auto point_batch = phi.quadrature_point(q);

                  for (unsigned int lane = 0;
                       lane < data.n_active_entries_per_cell_batch(cell_batch);
                       ++lane)
                    {
                      Point<dim, Number> quad_point;
                      for (unsigned int d = 0; d < dim; ++d)
                        quad_point[d] = point_batch[d][lane];

                      auto cp = closest_point_regular_grid<dim>(quad_point, n_points_tiff);
                      linear_index.emplace_back(cp.linear_index);
                    }
                }
            }
        },
        rhs,
        dummy,
        zero_out);

      // communicate indices to rank 0 who owns the properties
      // const auto indices_all =
      // dealii::Utilities::MPI::reduce<std::vector<std::size_t>>(
      // linear_index, mpi_comm, [](const auto &a, const auto &b) {
      // auto result = a;
      // result.insert(result.end(), b.begin(), b.end());
      // return result;
      //}, 0);

      const auto indices_all =
        dealii::Utilities::MPI::gather<std::vector<std::size_t>>(MPI_COMM_WORLD, linear_index, 0);
      // This will hold the properties for *this rank's* indices, in the same order as
      // `linear_index`
      std::vector<double> ghosted_properties(linear_index.size());
      const MPI_Comm      comm    = MPI_COMM_WORLD;
      const unsigned int  rank    = dealii::Utilities::MPI::this_mpi_process(comm);
      const unsigned int  n_ranks = dealii::Utilities::MPI::n_mpi_processes(comm);

      const int tag_props = 2001;

      if (rank == 0)
        {
          const std::vector<double> &values = properties;

          // keep futures if you want true nonblocking overlap
          std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
          send_futures.reserve(n_ranks > 0 ? n_ranks - 1 : 0);

          for (unsigned int p = 1; p < n_ranks; ++p)
            {
              const auto &idx_list = indices_all[p];

              std::vector<double> send_buffer;
              send_buffer.reserve(idx_list.size());
              for (auto idx : idx_list)
                send_buffer.push_back(values[idx]);

              send_futures.push_back(
                dealii::Utilities::MPI::isend(send_buffer, comm, p, tag_props));
            }

          // root fills its own
          const auto &idx0 = indices_all[0];
          ghosted_properties.resize(idx0.size());
          for (std::size_t i = 0; i < idx0.size(); ++i)
            ghosted_properties[i] = values[idx0[i]];

          // ensure sends completed (optional if futures will destruct at end of scope)
          for (auto &f : send_futures)
            f.wait();
        }
      else
        {
          auto fut = dealii::Utilities::MPI::irecv<std::vector<double>>(comm, 0, tag_props);

          ghosted_properties = fut.get(); // <-- actual receive happens here
          // (Optional sanity check)
          AssertThrow(ghosted_properties.size() == linear_index.size(),
                      dealii::ExcMessage("Received properties size mismatch."));
        }

      // 4) assign values
      unsigned int idx = 0;
      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &data, auto &dst, const auto & /*src*/, const auto cell_range) {
          FECellIntegrator phi(data);

          for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
               ++cell_batch)
            {
              phi.reinit(cell_batch);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  VectorizedArray<Number> val;

                  for (unsigned int lane = 0;
                       lane < data.n_active_entries_per_cell_batch(cell_batch);
                       ++lane)
                    {
                      val[lane] = ghosted_properties[idx];
                      idx += 1;
                    }
                  phi.submit_value(val, q);
                }

              phi.integrate_scatter(EvaluationFlags::values, rhs);
            }
        },
        rhs,
        dummy,
        zero_out);
    }
  else
    {
      // read the particle data on process 0
      auto data =
        get_particles_and_properties<dim, Number>(particle_file,
                                                  1 /*n_properties*/); //, " " /*delimiter*/);
      auto &particle_locations = data.first;
      {
        auto &properties_ = data.second; // properties [idx][comp]

        for (const auto &p : properties_)
          properties.emplace_back(p[0]);
      }
      // reduce points to bounding box
      // TODO: not hardcode DoFHandler index
      const auto &tria = matrix_free.get_dof_handler(0).get_triangulation();
      const auto &bb   = dealii::GridTools::compute_bounding_box(tria);

      // collect indices that are out of the bounding box
      std::vector<bool> remove_mask(properties.size(), false);
      for (unsigned int i = 0; i < data.first.size(); ++i)
        if (not bb.point_inside(data.first[i]))
          {
            remove_mask[i] = true;
          }

      {
        size_t write = 0;
        for (size_t read = 0; read < particle_locations.size(); ++read)
          {
            if (!remove_mask[read])
              particle_locations[write++] = std::move(particle_locations[read]);
          }
        particle_locations.resize(write);
      }

      {
        size_t write = 0;
        for (size_t read = 0; read < properties.size(); ++read)
          {
            if (!remove_mask[read])
              properties[write++] = std::move(properties[read]);
          }
        properties.resize(write);
      }

  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      print_entry(pcout, "Data points in bounding box: ", particle_locations.size());

      AssertThrow(properties.size() == particle_locations.size(),
                  dealii::ExcMessage(
                    "The number of properties and particle locations don't match"));

      if (enable_output)
        write_vtk_with_points<dim, Number>(particle_locations, properties, particle_output_file);

      // 1) collect quadrature points
      std::vector<Point<dim, Number>> quadrature_points;

      using VectorizedArrayType = VectorizedArray<Number>;
      using FECellIntegrator    = FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

      VectorType dummy;

      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &data, auto &dst, const auto & /*src*/, const auto cell_range) {
          FECellIntegrator phi(data);

          for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
               ++cell_batch)
            {
              phi.reinit(cell_batch);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  const auto point_batch = phi.quadrature_point(q);

                  for (unsigned int lane = 0;
                       lane < data.n_active_entries_per_cell_batch(cell_batch);
                       ++lane)
                    {
                      Point<dim, Number> quad_point;
                      for (unsigned int d = 0; d < dim; ++d)
                        quad_point[d] = point_batch[d][lane];

                      quadrature_points.emplace_back(quad_point);
                    }
                }
            }
        },
        rhs,
        dummy,
        zero_out);

      // 2) Perform distributed point search

      // Build ArborX distributed search tree using collected particle points
#ifdef DEAL_II_WITH_ARBORX
      dealii::ArborXWrappers::DistributedTree distributed_tree(MPI_COMM_WORLD, particle_locations);

      // Construct a nearest-neighbor search for each stencil point, looking for 1
      // nearest point
      dealii::ArborXWrappers::PointNearestPredicate bb_near(quadrature_points, 1);
      // Perform nearest neighbor search; returns matching local indices of particle
      // points and owning ranks
      const auto &[indices_and_ranks, offsets_stencil] = distributed_tree.query(bb_near);

      // Sanity check: ArborX gives 1 match per stencil point
      for (unsigned int i = 1; i < offsets_stencil.size(); ++i)
        Assert(offsets_stencil[i] - offsets_stencil[i - 1] == 1,
               dealii::ExcMessage("The offset needs to be one"));

      const auto [offset, n_global_particle_points] =
        dealii::Utilities::MPI::partial_and_total_sum(particle_locations.size(), MPI_COMM_WORLD);
      const auto offsets = dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, offset);

      // Global indices of particle points owned remotely (ghost points)
      // corresponding to the stencil point owned by this rank
      std::vector<dealii::types::global_dof_index> ghost_particle_indices;

      // Ghost indices (global) corresponding to matched particle points for each
      // locally owned stencil point
      for (const auto &[index, rank] : indices_and_ranks)
        ghost_particle_indices.push_back(offsets[rank] + index);

      // Noncontiguous partitioner used to communicate particle point values across
      // MPI ranks
      dealii::Utilities::MPI::NoncontiguousPartitioner partitioner;

      // Global indices of particle points owned by this MPI rank
      std::vector<dealii::types::global_dof_index> locally_owned_particle_indices;

      // Locally owned indices (global) of particle points owned by this process
      for (auto i = offset; i < offset + particle_locations.size(); ++i)
        locally_owned_particle_indices.push_back(i);

      // Set up communication pattern between local and ghost particle point data
      partitioner.reinit(locally_owned_particle_indices, ghost_particle_indices, MPI_COMM_WORLD);

      // Communicate each component array using the partitioner
      std::vector<double> ghosted_properties(ghost_particle_indices.size());

      partitioner.export_to_ghosted_array<double>(dealii::make_array_view(properties),
                                                  dealii::make_array_view(ghosted_properties));

      unsigned int idx = 0;

      AssertThrow(ghosted_properties.size() == quadrature_points.size(), ExcNotImplemented());
      // 3) assign values
      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &data, auto &dst, const auto & /*src*/, const auto cell_range) {
          FECellIntegrator phi(data);

          for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
               ++cell_batch)
            {
              phi.reinit(cell_batch);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  VectorizedArray<Number> val;

                  for (unsigned int lane = 0;
                       lane < data.n_active_entries_per_cell_batch(cell_batch);
                       ++lane)
                    {
                      val[lane] = ghosted_properties[idx];
                      idx += 1;
                    }
                  phi.submit_value(val, q);
                }

              phi.integrate_scatter(EvaluationFlags::values, rhs);
            }
        },
        rhs,
        dummy,
        zero_out);
#else
      AssertThrow(false,
                  dealii::ExcMessage("deal.II is not set up with Arborx, which is a prerequisite "
                                     "for using the \"quadrature_point_fast\" algorithm!"));
#endif
    }

  if (enable_memory_stats)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "create_rhs");
    }
}

}
