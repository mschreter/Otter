#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <multigrid.h>
#include <particle_util.h>
#include <screened_poisson_operator.h>
#include <sys/resource.h> // for rusage, getrusage, RUSAGE_SELF
#include <sys/time.h>     // sometimes also needed on some systems
#include <sys/types.h>
#include <utils.h>

#include <filesystem>
#include <memory>
#include <sstream>

#include "include/particle_util.h"
#include "include/utils.h"

template <int dim>
void
set_all_boundary_ids(Triangulation<dim> &tria, const types::boundary_id new_id)
{
  // Loop over all active cells
  for (const auto &cell : tria.active_cell_iterators())
    {
      // Loop over all faces of the cell
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
        {
          const auto face = cell->face(f);

          // Only modify boundary faces
          if (face->at_boundary())
            face->set_boundary_id(new_id);
        }
    }
}

struct Parameters
{
  // general settings
  unsigned int n_refinements                 = 5;
  unsigned int fe_degree                     = 1;
  double       screening_length              = 0;
  bool         neumann_bc                    = true;
  int          dim                           = 3;
  std::string  mesh_type                     = "";
  std::string  mesh_arguments                = "";
  std::string  particle_file                 = "";
  bool         particle_file_is_regular_grid = false;
  std::string  rhs_function                  = "";
  bool         enable_gmg                    = false;
  bool         use_simplex_mesh              = false;
  std::string  rhs_from_particle_formulation = "quadrature_point";

  // iterative solver (GMRES)
  unsigned int maxiter = 10000;
  double       abstol  = 1e-20;
  double       reltol  = 1e-10;

  // multigrid smoother
  std::string  smoother_type                = "chebyshev";
  double       smoother_smoothing_range     = 20;
  unsigned int smoother_degree              = 5;
  unsigned int smoother_eig_cg_n_iterations = 20;

  // multigrid coarse-grid solver
  std::string  coarse_solver_type            = "direct";
  unsigned int coarse_solver_maxiter         = 10000;
  double       coarse_solver_abstol          = 1e-20;
  double       coarse_solver_reltol          = 1e-4;
  unsigned int coarse_solver_smoother_sweeps = 1;
  unsigned int coarse_solver_n_cycles        = 1;
  std::string  coarse_solver_smoother_type   = "ILU";

  bool         compute_L2_norm_solution = true;
  bool         output_paraview          = true;
  bool         output_particles         = true;
  std::string  output_name              = "output";
  bool         enable_wall_times        = false;
  bool         output_memory            = false;
  unsigned int verbosity                = 0;

  void
  add_parameters(dealii::ParameterHandler &prm)
  {
    prm.add_parameter("verbosity",
                      verbosity,
                      "Verbosity. Choose 0 for tests and 1 for detailed output.");
    prm.add_parameter("dim", dim, "Set the dimension.");
    prm.add_parameter("n refinements", n_refinements, "Set the number of global refinements.");
    prm.add_parameter("screening length", screening_length, "Sets the screening length.");
    prm.add_parameter("mesh", mesh_type, "Sets the mesh type.");
    prm.add_parameter("mesh arguments",
                      mesh_arguments,
                      "Sets the arguments if a mesh is created using a grid generator.");
    prm.add_parameter("particle file", particle_file, "Sets the filename of the particles xyz.");
    prm.add_parameter("particle file is regular grid",
                      particle_file_is_regular_grid,
                      "Specify whether the particle file is a regular grid with spacing 1.");
    prm.add_parameter("rhs function", rhs_function, "Sets the RHS function.");
    prm.add_parameter("output name", output_name, "Sets the name of the output file.");
    prm.add_parameter("degree", fe_degree, "Set the finite element degree.");
    prm.add_parameter("enable gmg", enable_gmg, "Enable GMG or alternatively use AMG.");
    prm.add_parameter("use simplex mesh", use_simplex_mesh, "Use simplex mesh.");
    prm.add_parameter("output name", output_name, "Sets the name of the output file.");
    prm.add_parameter("neumann bc", neumann_bc, "Set to true.");
    prm.add_parameter("output memory", output_memory, "Enable output of memory consumbtion.");
    prm.add_parameter("enable wall time", enable_wall_times, "Enable wall times output.");
    prm.add_parameter("output paraview", output_paraview, "Enable Paraview output.");
    prm.add_parameter("compute L2 norm solution",
                      compute_L2_norm_solution,
                      "Compute L2 norm of the solution.");
    prm.add_parameter("output particles", output_particles, "Enable Paraview output.");
    prm.add_parameter("particle integration rhs",
                      rhs_from_particle_formulation,
                      "Compute error vs. reference solution.",
                      dealii::Patterns::Selection(
                        "quadrature_point_fast|quadrature_point|particle"));

    prm.enter_subsection("GMRES");
    {
      prm.add_parameter("max iterations", maxiter, "Maximum number of GMRES iterations.");
      prm.add_parameter("absolute tolerance", abstol, "Absolute convergence tolerance.");
      prm.add_parameter("relative tolerance", reltol, "Relative convergence tolerance.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Multigrid");
    {
      prm.add_parameter("smoother type", smoother_type, "Type of smoother (e.g., chebyshev).");
      prm.add_parameter("smoother smoothing range", smoother_smoothing_range, "Smoothing range.");
      prm.add_parameter("smoother degree", smoother_degree, "Degree of the smoother polynomial.");
      prm.add_parameter("smoother eigen CG iterations",
                        smoother_eig_cg_n_iterations,
                        "Number of CG iterations for estimating the largest eigenvalue.");
      prm.add_parameter("coarse solver type", coarse_solver_type, "Type of coarse grid solver.");
      prm.add_parameter("coarse solver max iterations",
                        coarse_solver_maxiter,
                        "Max CG iterations.");
      prm.add_parameter("coarse solver absolute tolerance",
                        coarse_solver_abstol,
                        "Absolute tolerance.");
      prm.add_parameter("coarse solver relative tolerance",
                        coarse_solver_reltol,
                        "Relative tolerance.");
      prm.add_parameter("coarse solver smoother sweeps",
                        coarse_solver_smoother_sweeps,
                        "Smoother sweeps.");
      prm.add_parameter("coarse solver V-cycle steps",
                        coarse_solver_n_cycles,
                        "Number of V-cycle steps.");
      prm.add_parameter("coarse solver smoother type",
                        coarse_solver_smoother_type,
                        "Smoother type (e.g., ILU).");
    }
    prm.leave_subsection();
  }
};

template <int dim, typename Number, typename NumberMG>
void
run(const Parameters &params, const std::filesystem::path &input_file_path)
{
  using VectorType       = LinearAlgebra::distributed::Vector<Number>;
  using VectorTypeMG     = LinearAlgebra::distributed::Vector<NumberMG>;
  using VectorTypeDouble = LinearAlgebra::distributed::Vector<double>;

  using MatrixType      = ScreenedPoissonOperator<dim, Number>;
  using LevelMatrixType = ScreenedPoissonOperator<dim, NumberMG>;

  using MGTransferType = MGTransferGlobalCoarsening<dim, VectorTypeMG>;

  const unsigned int min_level = 0;
  const unsigned int max_level = params.n_refinements;

  const MPI_Comm comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  print_banner(pcout);

  print_section(pcout, "Problem setup");

  print_entry(pcout, "Dimension", dim);
  print_entry(pcout, "Degree", params.fe_degree);

  if (params.verbosity > 0)
    print_entry(pcout, "MPI ranks", dealii::Utilities::MPI::n_mpi_processes(comm));

  TimerOutput timer(MPI_COMM_WORLD,
                    pcout,
                    params.enable_wall_times ? TimerOutput::summary : TimerOutput::never,
                    TimerOutput::wall_times);
  timer.enter_subsection("Create Triangulation");
  std::shared_ptr<Function<dim, Number>> dbc_func =
    std::make_shared<Functions::ConstantFunction<dim, Number>>(1.0);
  std::shared_ptr<Function<dim, Number>> rhs_func;

  if (params.rhs_function != "")
    {
      // If a csv file is provided as a rhs_function, assume that the rhs
      // represents a spherical particle packing
      if (ends_with(params.rhs_function, ".csv"))
        {
          const auto sphere_data = read_spherical_packing_data<dim>(params.rhs_function);
          rhs_func = std::make_shared<SphericalParticalPacking<dim, Number>>(sphere_data.first,
                                                                             sphere_data.second);
        }
      // Alternatively, an analytical function is given through the input file
      else
        {
          using Parser = dealii::FunctionParser<dim>; // IMPORTANT: match Number=double
          rhs_func     = std::make_shared<Parser>(params.rhs_function);
        }
    }

  std::shared_ptr<Triangulation<dim>> triangulation;

  if (dim == 1)
    {
      triangulation = std::make_shared<dealii::Triangulation<dim>>();
    }
  else if (params.use_simplex_mesh)
    {
      triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
        MPI_COMM_WORLD,
        typename Triangulation<dim>::MeshSmoothing(
          Triangulation<dim>::limit_level_difference_at_vertices),
        true,
        typename parallel::shared::Triangulation<dim>::Settings(
          parallel::shared::Triangulation<dim>::partition_zorder));
    }
  else
    {
      triangulation = std::make_shared<parallel::distributed::Triangulation<dim>>(
        MPI_COMM_WORLD,
        Triangulation<dim>::none,
        parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    }

  if (params.mesh_type == "hyper_cube" || params.mesh_type == "cylinder" ||
      params.mesh_type == "hyper_rectangle")
    {
      AssertThrow(not params.use_simplex_mesh,
                  ExcMessage("For grid generators, simplices are not supported."));
      GridGenerator::generate_from_name_and_arguments(*triangulation,
                                                      params.mesh_type,
                                                      params.mesh_arguments);
    }
  else
    {
      const auto mesh_file = input_file_path.parent_path() / params.mesh_type;

      GridIn<dim> grid_in(*triangulation);
      grid_in.read(mesh_file);
    }

  triangulation->refine_global(params.n_refinements);

  set_all_boundary_ids(*triangulation, 0);
  timer.leave_subsection();

  timer.enter_subsection("Setup DoF system");

  std::shared_ptr<Mapping<dim>>    mapping;
  std::shared_ptr<Quadrature<dim>> quadrature;
  DoFHandler<dim>                  dof_handler(*triangulation);

  if (params.use_simplex_mesh)
    {
      mapping =
        std::make_shared<dealii::MappingFE<dim>>(dealii::FE_SimplexP<dim>(params.fe_degree));
      quadrature = std::make_shared<dealii::QGaussSimplex<dim>>(params.fe_degree + 1);
      dof_handler.distribute_dofs(dealii::FE_SimplexP<dim>(params.fe_degree));
    }
  else
    {
      quadrature    = std::make_shared<dealii::QGauss<dim>>(params.fe_degree + 1);
      const auto fe = dealii::FE_Q<dim>(params.fe_degree);
      dof_handler.distribute_dofs(fe);
      mapping = std::make_shared<dealii::MappingQ<dim>>(params.fe_degree);
    }

  print_entry(pcout, "Number of Finite Elements", triangulation->n_cells());
  print_entry(pcout, "Number of DoFs", dof_handler.n_dofs());

  if (params.enable_gmg)
    dof_handler.distribute_mg_dofs();

  if (params.output_memory)
    {
      const double mem_dof =
        dealii::Utilities::MPI::sum(dof_handler.memory_consumption() / (1024.0 * 1024.0 * 1024),
                                    MPI_COMM_WORLD);
      const double mem_tria =
        dealii::Utilities::MPI::sum(triangulation->memory_consumption() / (1024.0 * 1024 * 1024),
                                    MPI_COMM_WORLD);
      print_entry(pcout, "MEMORY Triangulation (GB)", mem_tria);
      print_entry(pcout, "MEMORY DoFHandler (GB)", mem_dof);
    }

  AffineConstraints<Number> active_constraints;
  active_constraints.reinit(dof_handler.locally_owned_dofs(),
                            DoFTools::extract_locally_relevant_dofs(dof_handler));
  if (not params.neumann_bc)
    {
      active_constraints.reinit(dof_handler.locally_owned_dofs(),
                                DoFTools::extract_locally_relevant_dofs(dof_handler));
      VectorTools::interpolate_boundary_values(
        *mapping, dof_handler, 0, *dbc_func, active_constraints);
    }
  active_constraints.close();

  MatrixType active_operator(params.screening_length);
  active_operator.reinit(*mapping, dof_handler, active_constraints, *quadrature);

  VectorType solution, src;
  active_operator.initialize_dof_vector(solution);
  active_operator.initialize_dof_vector(src);



  timer.leave_subsection();

  Utilities::System::MemoryStats mem_before, mem_after, mem_after_solve;
  dealii::Utilities::System::get_memory_stats(mem_before);

  print_section(pcout, "Assembly of the right-hand side");

  if (rhs_func)
    {
      timer.enter_subsection("Create RHS");
      active_operator.rhs(src, rhs_func);
      timer.leave_subsection();
    }
  else
    {
      timer.enter_subsection("Create RHS");

      if constexpr (dim > 1)
        {
          const auto particle_file = input_file_path.parent_path() / params.particle_file;

          if (params.rhs_from_particle_formulation == "particle")
            {
              const auto output_file_name = params.output_name + "_particles.vtu";
              create_rhs_from_solid_particles<dim, Number, VectorType>(
                src,
                *mapping,
                *triangulation,
                particle_file.string(),
                active_operator.get_matrix_free(),
                output_file_name,
                params.output_particles,
                params.output_memory);
            }
          else if (params.rhs_from_particle_formulation == "quadrature_point")
            {
              const auto output_file_name = params.output_name + "_particles.vtk";
              create_rhs_from_solid_particles_closest_point<dim, Number, float, VectorType>(
                src,
                *mapping,
                particle_file.string(),
                active_operator.get_matrix_free(),
                output_file_name,
                params.output_particles,
                params.output_memory);
            }
          else if (params.rhs_from_particle_formulation == "quadrature_point_fast")
            {
              const auto output_file_name = params.output_name + "_particles.vtk";
              create_rhs_from_solid_particles_closest_point_fast<dim, Number, VectorType>(
                src,
                *mapping,
                particle_file.string(),
                active_operator.get_matrix_free(),
                output_file_name,
                params.particle_file_is_regular_grid,
                params.output_particles,
                params.output_memory);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }

      print_entry(pcout, "||RHS||l2", src.l2_norm());
      timer.leave_subsection();
    }

  if (params.output_memory)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "create_rhs");
    }

  // Finally, solve.
  ReductionControl solver_control(params.maxiter, params.abstol, params.reltol, false, false);

  if (params.enable_gmg)
    {
      timer.enter_subsection("Setup GMG");
      print_entry(pcout, "Preconditioner", "GMG");
      MGLevelObject<AffineConstraints<NumberMG>> mg_constraints(min_level, max_level);
      MGLevelObject<LevelMatrixType> mg_operators(min_level, max_level, params.screening_length);

      MGLevelObject<MGTwoLevelTransfer<dim, VectorTypeMG>> mg_transfers(min_level, max_level);

      MGConstrainedDoFs mg_constrained_dofs;
      mg_constrained_dofs.initialize(dof_handler);
      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, {0});

      // set up levels
      for (auto level = min_level; level <= max_level; ++level)
        {
          auto &constraints = mg_constraints[level];
          auto &op          = mg_operators[level];

          // set up mg_constraints
          constraints.reinit(dof_handler.locally_owned_mg_dofs(level),
                             DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

          mg_constrained_dofs.merge_constraints(constraints, level, true, false, true, true);

          constraints.close();

          // set up operator
          op.reinit(*mapping, dof_handler, constraints, *quadrature, level);
        }

      // set up transfer operator
      MGTransferGlobalCoarsening<dim, VectorTypeMG> mg_transfer(mg_constrained_dofs);
      mg_transfer.build(dof_handler, [&](const auto l, auto &vec) {
        mg_operators[l].initialize_dof_vector(vec);
      });

      using SmootherPreconditionerType = DiagonalMatrix<VectorTypeMG>;
      using SmootherType =
        PreconditionChebyshev<LevelMatrixType, VectorTypeMG, SmootherPreconditionerType>;
      using PreconditionerType = PreconditionMG<dim, VectorTypeMG, MGTransferType>;

      // Initialize level mg_operators.
      mg::Matrix<VectorTypeMG> mg_matrix(mg_operators);

      // Initialize smoothers.
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level, max_level);

      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          smoother_data[level].preconditioner = std::make_shared<SmootherPreconditionerType>();
          mg_operators[level].compute_inverse_diagonal(
            smoother_data[level].preconditioner->get_vector());
          smoother_data[level].smoothing_range     = params.smoother_smoothing_range;
          smoother_data[level].degree              = params.smoother_degree;
          smoother_data[level].eig_cg_n_iterations = params.smoother_eig_cg_n_iterations;
          smoother_data[level].constraints.copy_from(mg_constraints[level]);
        }

      MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorTypeMG> mg_smoother;
      mg_smoother.initialize(mg_operators, smoother_data);

      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          VectorTypeMG vec;
          mg_operators[level].initialize_dof_vector(vec);
          mg_smoother.smoothers[level].estimate_eigenvalues(vec);
        }

      // Initialize coarse-grid solver.
      ReductionControl              coarse_grid_solver_control(params.coarse_solver_maxiter,
                                                  params.coarse_solver_abstol,
                                                  params.coarse_solver_reltol,
                                                  false,
                                                  false);
      SolverGMRES<VectorTypeDouble> coarse_grid_solver(coarse_grid_solver_control);

      PreconditionIdentity precondition_identity;
      PreconditionChebyshev<LevelMatrixType, VectorTypeMG, DiagonalMatrix<VectorTypeMG>>
        precondition_chebyshev;

      TrilinosWrappers::PreconditionAMG precondition_amg;
      TrilinosWrappers::SolverDirect    precondition_direct;

      std::unique_ptr<MGCoarseGridBase<VectorTypeMG>> mg_coarse;

      if (params.coarse_solver_type == "gmres_with_amg")
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          amg_data.smoother_sweeps = params.coarse_solver_smoother_sweeps;
          amg_data.n_cycles        = params.coarse_solver_n_cycles;
          amg_data.smoother_type   = params.coarse_solver_smoother_type.c_str();

          // GMRES with AMG as preconditioner
          precondition_amg.initialize(mg_operators[min_level].get_system_matrix(), amg_data);

          mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<VectorTypeMG,
                                                                   SolverGMRES<VectorTypeDouble>,
                                                                   TrilinosWrappers::SparseMatrix,
                                                                   decltype(precondition_amg)>>(
            coarse_grid_solver, mg_operators[min_level].get_system_matrix(), precondition_amg);
        }
      else if ("direct")
        {
          precondition_direct.initialize(mg_operators[min_level].get_system_matrix());

          mg_coarse = std::make_unique<
            MGCoarseGridApplyPreconditioner<VectorTypeMG, TrilinosWrappers::SolverDirect>>(
            precondition_direct);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
      // Create multigrid object.
      Multigrid<VectorTypeMG> mg(
        mg_matrix, *mg_coarse, mg_transfer, mg_smoother, mg_smoother, min_level, max_level);

      // Convert it to a preconditioner.
      PreconditionerType preconditioner(dof_handler, mg, mg_transfer);
      timer.leave_subsection();

      timer.enter_subsection("Linear Solver");
      SolverCG<VectorType>(solver_control).solve(active_operator, solution, src, preconditioner);
      timer.leave_subsection();
    }
  else
    {
      timer.enter_subsection("Preconditioner");
      print_entry(pcout, "Preconditioner", "AMG");

      TrilinosWrappers::PreconditionAMG preconditioner;
      preconditioner.initialize(active_operator.get_system_matrix());

      // DiagonalMatrix<VectorType> preconditioner;
      // preconditioner.get_vector() = active_operator.get_diagonal();

      timer.leave_subsection();

      timer.enter_subsection("Linear Solver");

      print_section(pcout, "Linear Solver");
      SolverCG<VectorType>(solver_control).solve(active_operator, solution, src, preconditioner);

      timer.leave_subsection();
      print_entry(pcout, "Iterations:", solver_control.last_step());
    }
  active_constraints.distribute(solution);

  print_entry(pcout, "||u||l2", solution.l2_norm());

  if (params.output_memory)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_after, mem_after_solve, MPI_COMM_WORLD, "solve");
    }

  if (params.compute_L2_norm_solution)
    {
      // use a higher order quadrature rule for computing the L2 norm
      std::shared_ptr<Quadrature<dim>> quadrature_L2;
      if (params.use_simplex_mesh)
        quadrature_L2 = std::make_shared<dealii::QGaussSimplex<dim>>(
          std::min(static_cast<int>(params.fe_degree + 2), 4));
      else
        quadrature_L2 = std::make_shared<dealii::QGauss<dim>>(params.fe_degree + 2);

      Vector<Number> cell_wise_error(triangulation->n_active_cells());
      const auto     zero_func =
        std::make_shared<Functions::ZeroFunction<dim, Number>>(1 /*n components*/);

      solution.update_ghost_values();
      VectorTools::integrate_difference(*mapping,
                                        dof_handler,
                                        solution,
                                        *zero_func,
                                        cell_wise_error,
                                        *quadrature_L2,
                                        VectorTools::NormType::L2_norm);
      const auto error = VectorTools::compute_global_error(*triangulation,
                                                           cell_wise_error,
                                                           VectorTools::NormType::L2_norm);
      // pcout << "|solution|_L2 " << std::setw(15) << std::setprecision(15) << std::scientific
      //<< error << std::endl;

      print_section(pcout, "Post-Processing");

      print_entry(pcout, "||u||L2", error);
    }

  if (params.output_paraview)
    {
      timer.enter_subsection("Output");
      VectorType rhs_projected;
      active_operator.get_matrix_free().initialize_dof_vector(rhs_projected);

      project_vector<1, dim, VectorType>(
        *mapping, dof_handler, active_constraints, *quadrature, src, rhs_projected);

      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      if (dim > 1)
        {
          flags.write_higher_order_cells = true;
        }
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");

      VectorType voidRatio;
      active_operator.get_matrix_free().initialize_dof_vector(voidRatio);
      voidRatio = 1.;
      voidRatio -= solution;
      for (const auto &i : solution.locally_owned_elements())
        voidRatio[i] /= solution[i];

      data_out.add_data_vector(voidRatio, "voidRatio");
      data_out.add_data_vector(src, "rhs");
      data_out.add_data_vector(rhs_projected, "rhs_projected");

      Vector<Number> ranks(triangulation->n_active_cells());
      ranks = Utilities::MPI::this_mpi_process(comm);
      data_out.add_data_vector(ranks, "ranks");
      data_out.build_patches(*mapping, params.fe_degree);
      data_out.write_vtu_in_parallel(params.output_name + ".vtu", MPI_COMM_WORLD);
      timer.leave_subsection();
    }

  if (params.enable_wall_times)
    timer.print_wall_time_statistics(MPI_COMM_WORLD);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Utilities::System::MemoryStats mem_before, mem_after;
  dealii::Utilities::System::get_memory_stats(mem_before);
  Parameters params;

  auto print_help = [&]() {
    std::cout << "OTTER\n"
              << "-----\n\n"
              << "Usage:\n\n"
              << "  otter\n"
              << "      Run with built-in default parameters.\n\n"
              << "  otter input.json\n"
              << "      Read parameters from a JSON file.\n\n"
              << "Options:\n"
              << "  -h, --help\n"
              << "      Show this help message.\n\n";

    ParameterHandler prm;
    params.add_parameters(prm);

    std::cout << "Default parameters:\n";
    prm.print_parameters(std::cout, ParameterHandler::OutputStyle::ShortJSON);
  };

  namespace fs = std::filesystem;

  fs::path input_file_path;

  if (argc == 1)
    {
      // Run with built-in defaults.
    }
  else if (argc == 2)
    {
      const std::string arg(argv[1]);

      if (arg == "-h" || arg == "--help")
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            print_help();

          return 0;
        }

      input_file_path = fs::path(argv[1]);

      ParameterHandler prm;
      params.add_parameters(prm);

      std::ifstream file(input_file_path);

      AssertThrow(file.good(),
                  ExcMessage("Could not open parameter file <" + input_file_path.string() + ">."));

      prm.parse_input_from_json(file, true);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && params.verbosity > 0)
        prm.print_parameters(std::cout, ParameterHandler::OutputStyle::ShortJSON);
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("Wrong run command.\n"
                             "Use one of:\n"
                             "  otter\n"
                             "  otter input.json\n"
                             "  otter --help"));
    }

  if (params.dim == 1)
    run<1, double, float>(params, input_file_path);
  else if (params.dim == 2)
    run<2, double, float>(params, input_file_path);
  else if (params.dim == 3)
    run<3, double, float>(params, input_file_path);
  else
    {
      AssertThrow(false, ExcMessage("Your run command is wrong."));
    }

  if (params.output_memory)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "main");
    }
}
