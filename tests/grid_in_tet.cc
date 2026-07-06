#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>

#include <particle_util.h>
#include <utils.h>

#include <fstream>
#include <iostream>
#include <string>

template <int dim, typename Number>
void
test(std::string filename, std::string particle_file)
{
  using namespace Otter;
  using namespace dealii;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;
  using VectorizedArrayType = VectorizedArray<Number>;
  using FECellIntegrator    = FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  int fe_degree              = 1;
  int max_particles_per_cell = 25;
  int n_refinements          = 1;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);

  GridIn<dim> grid_in(triangulation);


  std::ifstream in_file(filename);
  grid_in.read_abaqus(in_file);

  triangulation.refine_global(n_refinements);

  MappingQ<dim> mapping(2);

  // setup matrix-free
  QGauss<dim> quadrature(fe_degree + 1);
  FE_Q<dim>   fe(fe_degree);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;
  constraints.close();

  MatrixFree<dim, Number, VectorizedArray<Number>> matrix_free;
  {
    typename MatrixFree<dim, Number, VectorizedArray<Number>>::AdditionalData data;
    data.mapping_update_flags =
      update_quadrature_points | update_gradients | update_values | update_JxW_values;

    matrix_free.reinit(mapping, dof_handler, constraints, quadrature, data);
  }

  VectorType rhs;
  VectorType dummy;
  matrix_free.initialize_dof_vector(rhs);

  // APPROACH 1:
  dealii::Particles::ParticleHandler<dim> particle_handler;
  create_rhs_from_solid_particles<dim, Number, VectorType>(
    rhs, mapping, triangulation, particle_file, matrix_free, "particle.vtu", false, false);

  // APPROACH 2:
  if (false)
    {
      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &data, auto &dst, const auto & /*src*/, const auto cell_range) {
          FECellIntegrator            fe_eval(data);
          FEPointEvaluation<dim, dim> fe_point_eval(mapping, fe, update_values);

          for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
               ++cell_batch)
            {
              VectorizedArrayType submit = VectorizedArrayType(0);

              for (unsigned int lane = 0;
                   lane < data.n_active_entries_per_cell_batch(fe_eval.get_current_cell_index());
                   ++lane)
                {
                  // OLD
                  const int n_particles_in_cell = particle_handler.n_particles_in_cell(
                    matrix_free.get_cell_iterator(cell_batch, lane));
                  std::cout << "n particles in cell: " << n_particles_in_cell << std::endl;

                  if (n_particles_in_cell > 0)
                    submit[lane] = n_particles_in_cell / max_particles_per_cell;
                }

              for (unsigned int q : fe_eval.quadrature_point_indices())
                fe_eval.submit_value(submit, q);

              fe_eval.integrate_scatter(dealii::EvaluationFlags::values, dst);
            }
        },
        rhs,
        dummy,
        true);
    }

  // debug output
  if (false)
    {
      dealii::Particles::DataOut<dim, dim> particle_output;
      std::vector<std::string>             solution_names(1, "value");
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        particle_data_component_interpretation(
          1, dealii::DataComponentInterpretation::component_is_scalar);
      particle_output.build_patches(particle_handler,
                                    solution_names,
                                    particle_data_component_interpretation);

      particle_output.write_vtu_in_parallel("particle.vtu", MPI_COMM_WORLD);

      VectorType rhs_projected;
      matrix_free.initialize_dof_vector(rhs_projected);

      project_vector<1, dim, VectorType>(
        mapping, dof_handler, constraints, quadrature, rhs, rhs_projected);

      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(rhs, "rhs");
      data_out.add_data_vector(rhs_projected, "rhs_projected");

      data_out.build_patches(mapping,
                             fe_degree + 1,
                             DataOut<dim>::CurvedCellRegion::curved_inner_cells);
      data_out.write_vtu_in_parallel("cells.vtu", MPI_COMM_WORLD);
    }
}

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test<3, double>(SOURCE_DIR "/mesh/mesh_hex.inp", SOURCE_DIR "/points/allPoints_sampleFac100.xyz");
}
