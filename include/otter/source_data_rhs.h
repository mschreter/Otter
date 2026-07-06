
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
#include <otter/read_tiff.h>
#include <otter/utils.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>


namespace Otter
{

  /**
   * @brief Read source points and associated source values from a file.
   *
   * Reads source data on MPI rank 0 from either an `.xyz` text file or a
   * `.tif` image file. For `.xyz` files, each data line is expected to contain
   * `dim` coordinate values followed by `n_source_values` scalar source values,
   * separated by `delimiter`. For `.tif` files, the source points and
   * source values are obtained through `read_tiff()`.
   *
   * @tparam dim Spatial dimension.
   * @tparam Number Scalar type used for coordinates and source values.
   *
   * @param[in] source_data_file Path to the source data input file. Supported
   * formats are `.xyz` and `.tif`.
   * @param[in] n_source_values Number of scalar source values stored per source point.
   * @param[in] delimiter Field delimiter used for `.xyz` files.
   * @param[in] mpi_communicator MPI communicator used to identify rank 0.
   *
   * @return A pair containing the source points and a vector of source-value
   * vectors. The returned data is populated on rank 0.
   *
   * @throws dealii::ExcMessage If the input file cannot be opened or if the file
   * extension is unsupported.
   */
  template <int dim, typename Number>
  inline std::pair<std::vector<dealii::Point<dim, Number>>, std::vector<std::vector<Number>>>
  read_source_data(std::string    source_data_file,
                               unsigned int   n_source_values,
                               char           delimiter        = ' ',
                               const MPI_Comm mpi_communicator = MPI_COMM_WORLD)
  {
    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                                       0);
    // Make global source-value vector
    std::vector<dealii::Point<dim, Number>> source_points;
    std::vector<std::vector<Number>>        source_values{};

    // serial read of xyz
    // TODO: perform read in parallel
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        if (source_data_file.ends_with(".xyz"))
          {
            std::fstream file;
            file.open(source_data_file, std::ios::in);
            AssertThrow(!(file.fail()),
                        dealii::ExcMessage("Unable to open source data file \"" +
                                           source_data_file + "\". Aborting!"));
            std::string line;
            std::getline(file, line); // Ignore the first line
            while (std::getline(file, line))
              {
                std::vector<Number> source_value_vector(n_source_values, 0.0);
                std::string         temp;
                std::istringstream  data_string(line);
                // source point
                dealii::Point<dim, Number> location;
                for (unsigned int i = 0; i < dim; i++)
                  {
                    std::getline(data_string, temp, delimiter);
                    location[i] = std::stod(temp);
                  }
                source_points.push_back(location);
                for (unsigned int i = 0; i < n_source_values; i++)
                  {
                    std::getline(data_string, temp, delimiter);
                    source_value_vector[i] = std::stod(temp);
                  }
                source_values.push_back(source_value_vector);
              }
          }
        else if (source_data_file.ends_with(".tif"))
          {
            // TODO: add parameters for scaling and shift
            auto data = read_tiff<dim, Number>(source_data_file, 1, 1, 1, 0, 0, 0);
            source_points.swap(data.first);
            source_values.swap(data.second);
          }
        else
          AssertThrow(false, dealii::ExcMessage("Only *.xyz or *.tif files are supported."));

        print_entry(pcout, "Source values read", source_values.size());
      }

    return {source_points, source_values};
  }

  /**
   * @brief Initialize the source-point handler from an input file.
   *
   * This function reads source data from a specified input file and initializes
   * the given `source_point_handler` with the provided source points and
   * associated source values. The source points are inserted into the domain
   * defined by the provided triangulation and mapping.
   *
   * The number of source values per source point is specified explicitly, and the
   * function will read that many values for each source point from the file. Each
   * line in the input file should contain `dim` position coordinates followed by
   * `n_source_values` source values, separated by the specified delimiter.
   *
   * @tparam dim The spatial dimension of the problem.
   * @tparam Number The numeric type used for source-point coordinates and source values.
   *
   * @param[out] source_point_handler The source-point handler to be populated with
   * source data.
   * @param[in] triangulation The triangulation defining the spatial domain.
   * @param[in] mapping The mapping used for placing source points in the curved
   * domain.
   * @param[in] source_data_file Path to the input file containing source data.
   * @param[in] n_source_values The Number of source values associated with each
   * source point.
   * @param[in] delimiter The character used to separate fields in the data file
   * (default is space).
   * @param[in] mpi_communicator The MPI communicator used for parallel
   * distribution (default is MPI_COMM_WORLD).
   */
  template <int dim, typename Number>
  inline void
  initialize_source_point_handler(dealii::Particles::ParticleHandler<dim> &source_point_handler,
                              const dealii::Triangulation<dim>        &triangulation,
                              const dealii::Mapping<dim>              &mapping,
                              std::string                              source_data_file,
                              unsigned int                             n_source_values,
                              char                                     delimiter = ' ',
                              const MPI_Comm mpi_communicator                    = MPI_COMM_WORLD)
  {
    const auto data =
      read_source_data<dim, Number>(source_data_file, n_source_values, delimiter);

    const auto &source_points = data.first;
    const auto &source_values         = data.second;

    // Insert the source points into the deal.II particle handler.
    std::vector<dealii::BoundingBox<dim>> local_bounding_box =
      dealii::GridTools::compute_mesh_predicate_bounding_box(
        triangulation, dealii::IteratorFilters::LocallyOwnedCell());
    std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_box =
      dealii::Utilities::MPI::all_gather(mpi_communicator, local_bounding_box);

    source_point_handler.initialize(triangulation, mapping, n_source_values);

    source_point_handler.insert_global_particles(dealii::Utilities::MPI::this_mpi_process(
                                               mpi_communicator) == 0 ?
                                               source_points :
                                               std::vector<dealii::Point<dim, Number>>{},
                                             global_bounding_box,
                                             source_values);
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      std::cout << "Source points global insertion finished: " << source_point_handler.n_global_particles()
                << std::endl;
  }

  /**
   * @brief Assemble a right-hand side vector from source points located inside
   * finite elements.
   *
   * Reads source points and scalar source values from a source data file,
   * initializes a deal.II particle handler, and distributes the source values
   * onto the finite element space by evaluating shape functions at the source
   * point locations. The resulting values are integrated into the output vector.
   *
   * @tparam dim Spatial dimension.
   * @tparam Number Scalar type used by the matrix-free operator.
   * @tparam VectorType Type of the output vector.
   *
   * @param[out] rhs Right-hand side vector to assemble.
   * @param[in] mapping Mapping used for point evaluation in physical cells.
   * @param[in] triangulation The mesh over which the source points are defined.
   * @param[in] source_data_file Path to the source data input file.
   * @param[in] matrix_free Matrix-free data used for cell-wise integration.
   * @param[in] source_output_file Path for optional source-data output.
   * @param[in] enable_output If true, writes the source data to VTK output.
   * @param[in] enable_memory_stats If true, prints memory statistics.
   * @param[in] zero_out If true, zeroes `rhs` before assembly.
   */
  template <int dim, typename Number, typename VectorType>
  inline void
  assemble_rhs_from_source_point_quadrature(
    VectorType                                                             &rhs,
    const dealii::Mapping<dim>                                             &mapping,
    const dealii::Triangulation<dim>                                       &triangulation,
    std::string                                                             source_data_file,
    const dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> &matrix_free,
    std::string                                                             source_output_file,
    const bool                                                              enable_output,
    const bool                                                              enable_memory_stats,
    const bool                                                              zero_out = true)
  {
    using namespace dealii;

    Utilities::System::MemoryStats mem_before, mem_after;
    dealii::Utilities::System::get_memory_stats(mem_before);

    using VectorizedArrayType = VectorizedArray<Number>;
    dealii::Particles::ParticleHandler<dim> source_point_handler;

    initialize_source_point_handler<dim, Number>(source_point_handler,
                                             triangulation,
                                             mapping,
                                             source_data_file,
                                             1 /*n_components*/,
                                             ' ' /*delimiter*/);

    VectorType dummy;

    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &data, auto &dst, const auto & /*src*/, const auto cell_range) {
        FEPointEvaluation<1, dim, dim, Number> fe_point_eval(mapping,
                                                             data.get_dof_handler().get_fe(),
                                                             update_values);

        std::vector<Number> local_values(data.get_dof_handler().get_fe().n_dofs_per_cell());
        std::vector<types::global_dof_index> local_dof_indices(
          data.get_dof_handler().get_fe().n_dofs_per_cell());

        for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
             ++cell_batch)
          {
            for (unsigned int lane = 0; lane < data.n_active_entries_per_cell_batch(cell_batch);
                 ++lane)
              {
                const auto cell_volume = data.get_cell_iterator(cell_batch, lane)->measure();
                data.get_cell_iterator(cell_batch, lane)
                  ->as_dof_handler_iterator(data.get_dof_handler())
                  ->get_dof_indices(local_dof_indices);

                auto n_points =
                  source_point_handler.n_particles_in_cell(data.get_cell_iterator(cell_batch, lane));

                if (n_points == 0)
                  continue;

                std::vector<Point<dim, Number>> unit_points;
                std::vector<Number>             val;
                for (const auto &p :
                     source_point_handler.particles_in_cell(data.get_cell_iterator(cell_batch, lane)))
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
          sample_data_component_interpretation(
            1, dealii::DataComponentInterpretation::component_is_scalar);
        dealii::Particles::DataOut<dim, dim> source_data_output;
        source_data_output.build_patches(source_point_handler,
                                      solution_names,
                                      sample_data_component_interpretation);

        source_data_output.write_vtu_in_parallel(source_output_file, MPI_COMM_WORLD);
      }

    if (enable_memory_stats)
      {
        dealii::Utilities::System::get_memory_stats(mem_after);
        print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "create_rhs");
      }
  }

  /**
   * @brief Assemble a right-hand side vector by assigning values from the
   * nearest source point to each quadrature point.
   *
   * Reads source points and scalar source values on rank 0,
   * broadcasts the data to all MPI ranks, builds a nearest-neighbor search tree,
   * and evaluates the closest source point value at every quadrature point of the
   * matrix-free finite element space. The resulting values are integrated into
   * the output vector.
   *
   * This implementation replicates the source data on all ranks and performs
   * the closest-point search locally.
   *
   * @tparam dim Spatial dimension.
   * @tparam Number Scalar type used by the matrix-free operator.
   * @tparam NumberCPP Scalar type used for source-point storage and point searches.
   * @tparam VectorType Type of the output vector.
   *
   * @param[out] rhs Right-hand side vector to assemble.
   * @param[in] mapping Mapping object. Currently unused by this implementation.
   * @param[in] source_data_file Path to the source data input file.
   * @param[in] matrix_free Matrix-free data used for cell-wise integration.
   * @param[in] source_output_file Path for optional source-data output.
   * @param[in] enable_output If true, writes the source data to VTK output.
   * @param[in] enable_memory_stats If true, prints memory statistics.
   * @param[in] zero_out If true, zeroes `rhs` before assembly.
   */
  template <int dim, typename Number, typename NumberCPP, typename VectorType>
  inline void
  assemble_rhs_from_standard_quadrature(
    VectorType &rhs,
    const dealii::Mapping<dim> & /*mapping*/,
    std::string                                                             source_data_file,
    const dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> &matrix_free,
    std::string                                                             source_output_file,
    const bool                                                              enable_output,
    const bool                                                              enable_memory_stats,
    const bool                                                              zero_out = true)
  {
    using namespace dealii;
    Utilities::System::MemoryStats mem_before, mem_after, mem_after_ptree;
    dealii::Utilities::System::get_memory_stats(mem_before);

    // read the source data on process 0
    const auto data =
      read_source_data<dim, NumberCPP>(source_data_file,
                                                   1 /*n_source_values*/); //, " " /*delimiter*/);

    auto source_points = data.first;
    auto source_values         = data.second;

    if (enable_output)
      write_vtk_with_points<dim, NumberCPP>(source_points, source_values, source_output_file);

    // distribute source data to all processes
    source_points = Utilities::MPI::broadcast(MPI_COMM_WORLD, source_points, 0);

    source_values = Utilities::MPI::broadcast(MPI_COMM_WORLD, source_values, 0);

    using ContainerType = std::vector<BoundingBox<dim, NumberCPP>>;

    ContainerType source_point_boxes;
    for (const auto &p : source_points)
      source_point_boxes.emplace_back(BoundingBox<dim, NumberCPP>(p));

    const auto source_point_tree = pack_rtree_of_indices(source_point_boxes);

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

        for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
             ++cell_batch)
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

                    std::vector<typename ContainerType::size_type> closest_source_point;
                    source_point_tree.query(boost::geometry::index::nearest(quad_point, 1),
                                         std::back_inserter(closest_source_point));

                    if (closest_source_point.size() != 1)
                      {
                        std::cout << "WARNING: The number of found nearest points is "
                                     "wrong. We found"
                                  << closest_source_point.size() << " points. The quadrature point is ";
                        for (unsigned int d = 0; d < dim; ++d)
                          std::cout << quad_point[d] << " ";

                        std::cout << std::endl;
                      }

                    val[lane] = source_values[closest_source_point[0]][0];
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

  /**
   * @brief Assemble a right-hand side vector using a memory-efficient
   * closest-point projection from source data.
   *
   * Assigns scalar source data to finite element quadrature points by
   * evaluating the closest source point value and integrating the resulting field
   * into `rhs`. Two execution paths are supported:
   *
   * - For regular-grid TIFF input, only the scalar image data is stored on rank 0.
   *   Each rank computes the closest regular-grid index for its local quadrature
   *   points, sends those indices to rank 0, and receives the corresponding
   *   source values.
   * - For general source data, source points are filtered to the mesh
   *   bounding box and a distributed ArborX nearest-neighbor search is used.
   *
   * The general source-data path requires deal.II to be configured with
   * ArborX support.
   *
   * @tparam dim Spatial dimension.
   * @tparam Number Scalar type used by the matrix-free operator.
   * @tparam VectorType Type of the output vector.
   *
   * @param[out] rhs Right-hand side vector to assemble.
   * @param[in] mapping Mapping used by the matrix-free finite element space.
   * @param[in] source_data_file Path to the source data input file.
   * @param[in] matrix_free Matrix-free data used for quadrature-point traversal
   * and cell-wise integration.
   * @param[in] source_output_file Path for optional source-data output.
   * @param[in] is_regular_grid If true, use the memory-efficient regular-grid
   * TIFF path.
   * @param[in] enable_output If true, writes the input source data to
   * VTK output.
   * @param[in] enable_memory_stats If true, prints memory statistics.
   * @param[in] zero_out If true, zeroes `rhs` before assembly.
   *
   * @throws dealii::ExcMessage If `is_regular_grid` is true but the input file is
   * not a TIFF file.
   * @throws dealii::ExcMessage If ArborX support is required but deal.II was not
   * configured with ArborX.
   */
  template <int dim, typename Number, typename VectorType>
  inline void
  assemble_rhs_from_standard_quadrature_fast(
    VectorType                                                             &rhs,
    const dealii::Mapping<dim>                                             &mapping,
    std::string                                                             source_data_file,
    const dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> &matrix_free,
    std::string                                                             source_output_file,
    const bool                                                              is_regular_grid,
    const bool                                                              enable_output,
    const bool                                                              enable_memory_stats,
    const bool                                                              zero_out = true)
  {
    using namespace dealii;
    Utilities::System::MemoryStats mem_before, mem_after;
    dealii::Utilities::System::get_memory_stats(mem_before);

    std::vector<double> source_values;
    if (is_regular_grid)
      {
        AssertThrow(source_data_file.ends_with(".tif"),
                    ExcMessage("this option is only supported for tiff files."));
        // 1) read source values from tiff file
        std::array<int, dim> voxel_dimensions;
        if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::cout << "MEMORY EFFICIENT PATH: store only data from tiff file" << std::endl;
            const auto &data = read_tiff_data_only<dim>(source_data_file);
            voxel_dimensions    = data.second;
            source_values       = data.first;
            // communicate the number of points to all processes
          }

        voxel_dimensions = Utilities::MPI::broadcast(MPI_COMM_WORLD, voxel_dimensions, 0);

        // 2) collect quadrature points and perform closest point search
        std::vector<std::size_t> voxel_indices; // row-major linear index (x fastest)

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

                        auto cp = closest_point_regular_grid<dim>(quad_point, voxel_dimensions);
                        voxel_indices.emplace_back(cp.linear_index);
                      }
                  }
              }
          },
          rhs,
          dummy,
          zero_out);

        const auto voxel_indices_all =
          dealii::Utilities::MPI::gather<std::vector<std::size_t>>(MPI_COMM_WORLD, voxel_indices, 0);
        // This will hold the source values for *this rank's* indices, in the same order as
        // `voxel_indices`
        std::vector<double> ghosted_source_values(voxel_indices.size());
        const MPI_Comm      comm    = MPI_COMM_WORLD;
        const unsigned int  rank    = dealii::Utilities::MPI::this_mpi_process(comm);
        const unsigned int  n_ranks = dealii::Utilities::MPI::n_mpi_processes(comm);

        const int tag_source_values = 2001;

        if (rank == 0)
          {
            const std::vector<double> &values = source_values;

            // keep futures if you want true nonblocking overlap
            std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
            send_futures.reserve(n_ranks > 0 ? n_ranks - 1 : 0);

            for (unsigned int p = 1; p < n_ranks; ++p)
              {
                const auto &idx_list = voxel_indices_all[p];

                std::vector<double> send_buffer;
                send_buffer.reserve(idx_list.size());
                for (auto idx : idx_list)
                  send_buffer.push_back(values[idx]);

                send_futures.push_back(
                  dealii::Utilities::MPI::isend(send_buffer, comm, p, tag_source_values));
              }

            // root fills its own
            const auto &idx0 = voxel_indices_all[0];
            ghosted_source_values.resize(idx0.size());
            for (std::size_t i = 0; i < idx0.size(); ++i)
              ghosted_source_values[i] = values[idx0[i]];

            // ensure sends completed (optional if futures will destruct at end of scope)
            for (auto &f : send_futures)
              f.wait();
          }
        else
          {
            auto fut = dealii::Utilities::MPI::irecv<std::vector<double>>(comm, 0, tag_source_values);

            ghosted_source_values = fut.get(); // <-- actual receive happens here
            // (Optional sanity check)
            AssertThrow(ghosted_source_values.size() == voxel_indices.size(),
                        dealii::ExcMessage("Received source-value size mismatch."));
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
                        val[lane] = ghosted_source_values[idx];
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
        // read the source data on process 0
        auto data =
          read_source_data<dim, Number>(source_data_file,
                                                    1 /*n_source_values*/); //, " " /*delimiter*/);
        auto &source_points = data.first;
        {
          auto &source_values_ = data.second; // source_values[idx][comp]

          for (const auto &p : source_values_)
            source_values.emplace_back(p[0]);
        }
        // reduce points to bounding box
        // TODO: not hardcode DoFHandler index
        const auto &tria = matrix_free.get_dof_handler(0).get_triangulation();
        const auto &bb   = dealii::GridTools::compute_bounding_box(tria);

        // collect indices that are out of the bounding box
        std::vector<bool> remove_mask(source_values.size(), false);
        for (unsigned int i = 0; i < data.first.size(); ++i)
          if (not bb.point_inside(data.first[i]))
            {
              remove_mask[i] = true;
            }

        {
          size_t write = 0;
          for (size_t read = 0; read < source_points.size(); ++read)
            {
              if (!remove_mask[read])
                source_points[write++] = std::move(source_points[read]);
            }
          source_points.resize(write);
        }

        {
          size_t write = 0;
          for (size_t read = 0; read < source_values.size(); ++read)
            {
              if (!remove_mask[read])
                source_values[write++] = std::move(source_values[read]);
            }
          source_values.resize(write);
        }

        dealii::ConditionalOStream pcout(std::cout,
                                         dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                                           0);
        print_entry(pcout, "Data points in bounding box: ", source_points.size());

        AssertThrow(source_values.size() == source_points.size(),
                    dealii::ExcMessage(
                      "The number of source values and source points do not match"));

        if (enable_output)
          write_vtk_with_points<dim, Number>(source_points, source_values, source_output_file);

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

        // Build ArborX distributed search tree using collected source points
#ifdef DEAL_II_WITH_ARBORX
        dealii::ArborXWrappers::DistributedTree distributed_tree(MPI_COMM_WORLD,
                                                                 source_points);

        // Construct a nearest-neighbor search for each quadrature point, looking for 1
        // nearest point
        dealii::ArborXWrappers::PointNearestPredicate bb_near(quadrature_points, 1);
        // Perform nearest neighbor search; returns matching local indices of source points and owning ranks
        const auto &[indices_and_ranks, offsets_stencil] = distributed_tree.query(bb_near);

        // Sanity check: ArborX gives 1 match per quadrature point
        for (unsigned int i = 1; i < offsets_stencil.size(); ++i)
          Assert(offsets_stencil[i] - offsets_stencil[i - 1] == 1,
                 dealii::ExcMessage("The offset needs to be one"));

        const auto [offset, n_global_source_points] =
          dealii::Utilities::MPI::partial_and_total_sum(source_points.size(), MPI_COMM_WORLD);
        const auto offsets = dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, offset);

        // Global indices of source points owned remotely (ghost points)
        // corresponding to the quadrature point owned by this rank
        std::vector<dealii::types::global_dof_index> ghost_source_point_indices;

        // Ghost indices (global) corresponding to matched source points for each
        // locally owned quadrature point
        for (const auto &[index, rank] : indices_and_ranks)
          ghost_source_point_indices.push_back(offsets[rank] + index);

        // Noncontiguous partitioner used to communicate source point values across
        // MPI ranks
        dealii::Utilities::MPI::NoncontiguousPartitioner partitioner;

        // Global indices of source points owned by this MPI rank
        std::vector<dealii::types::global_dof_index> locally_owned_source_point_indices;

        // Locally owned indices (global) of source points owned by this process
        for (auto i = offset; i < offset + source_points.size(); ++i)
          locally_owned_source_point_indices.push_back(i);

        // Set up communication pattern between local and ghost source point data
        partitioner.reinit(locally_owned_source_point_indices, ghost_source_point_indices, MPI_COMM_WORLD);

        // Communicate each component array using the partitioner
        std::vector<double> ghosted_source_values(ghost_source_point_indices.size());

        partitioner.export_to_ghosted_array<double>(dealii::make_array_view(source_values),
                                                    dealii::make_array_view(ghosted_source_values));

        unsigned int idx = 0;

        AssertThrow(ghosted_source_values.size() == quadrature_points.size(), ExcNotImplemented());
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
                        val[lane] = ghosted_source_values[idx];
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
} // namespace Otter
