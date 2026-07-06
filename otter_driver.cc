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

#include <otter/multigrid.h>
#include <otter/parameters.h>
#include <otter/screened_poisson_operator.h>
#include <otter/source_data_rhs.h>
#include <otter/utils.h>
#include <sys/resource.h> // for rusage, getrusage, RUSAGE_SELF
#include <sys/time.h>     // sometimes also needed on some systems
#include <sys/types.h>

#include <filesystem>

using namespace Otter;

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

template <int dim, typename Number, typename NumberMG>
void
run(const ScreenedPoissonParameters &params, const std::filesystem::path &input_file_path)
{
  using VectorType       = LinearAlgebra::distributed::Vector<Number>;
  using VectorTypeMG     = LinearAlgebra::distributed::Vector<NumberMG>;
  using VectorTypeDouble = LinearAlgebra::distributed::Vector<double>;

  using MatrixType      = ScreenedPoissonOperator<dim, Number>;
  using LevelMatrixType = ScreenedPoissonOperator<dim, NumberMG>;

  using MGTransferType = MGTransferGlobalCoarsening<dim, VectorTypeMG>;

  const unsigned int min_level = 0;
  const unsigned int max_level = params.mesh.n_global_refinements;

  const MPI_Comm comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  print_banner(pcout);

  print_section(pcout, "Problem setup");

  print_entry(pcout, "Dimension", dim);
  print_entry(pcout, "Degree", params.finite_element.degree);

  if (params.output.verbosity > 0)
    print_entry(pcout, "MPI ranks", dealii::Utilities::MPI::n_mpi_processes(comm));

  TimerOutput timer(MPI_COMM_WORLD,
                    pcout,
                    params.output.print_wall_times ? TimerOutput::summary : TimerOutput::never,
                    TimerOutput::wall_times);
  timer.enter_subsection("Create Triangulation");
  std::shared_ptr<Function<dim, Number>> dbc_func =
    std::make_shared<Functions::ConstantFunction<dim, Number>>(1.0);
  std::shared_ptr<Function<dim, Number>> rhs_func;

  if (params.source_data.analytical_function != "")
    {
      // If a csv file is provided as the analytical function, assume that it
      // represents a spherical packing.
      if (ends_with(params.source_data.analytical_function, ".csv"))
        {
          const auto sphere_data =
            read_spherical_packing_data<dim>(params.source_data.analytical_function);
          rhs_func = std::make_shared<SphericalParticalPacking<dim, Number>>(sphere_data.first,
                                                                             sphere_data.second);
        }
      // Alternatively, parse an analytical function given through the input file.
      else
        {
          using Parser = dealii::FunctionParser<dim>; // IMPORTANT: match Number=double
          rhs_func     = std::make_shared<Parser>(params.source_data.analytical_function);
        }
    }

  std::shared_ptr<Triangulation<dim>> triangulation;

  if (dim == 1)
    {
      triangulation = std::make_shared<dealii::Triangulation<dim>>();
    }
  else if (params.mesh.use_simplex_mesh)
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

  if (params.mesh.type == "hyper_cube" || params.mesh.type == "cylinder" ||
      params.mesh.type == "hyper_rectangle")
    {
      AssertThrow(not params.mesh.use_simplex_mesh,
                  ExcMessage("For grid generators, simplices are not supported."));
      GridGenerator::generate_from_name_and_arguments(*triangulation,
                                                      params.mesh.type,
                                                      params.mesh.arguments);
    }
  else
    {
      const auto    mesh_file = input_file_path.parent_path() / params.mesh.type;
      std::ifstream in(mesh_file);
      AssertThrow(in, ExcMessage("Could not open mesh file: " + mesh_file.string()));


      GridIn<dim> grid_in(*triangulation);

      if (mesh_file.extension() == ".inp")
        grid_in.read_abaqus(in);
      else
        grid_in.read(mesh_file);
    }

  triangulation->refine_global(params.mesh.n_global_refinements);

  set_all_boundary_ids(*triangulation, 0);
  timer.leave_subsection();

  timer.enter_subsection("Setup DoF system");

  std::shared_ptr<Mapping<dim>>    mapping;
  std::shared_ptr<Quadrature<dim>> quadrature;
  DoFHandler<dim>                  dof_handler(*triangulation);

  if (params.mesh.use_simplex_mesh)
    {
      mapping = std::make_shared<dealii::MappingFE<dim>>(
        dealii::FE_SimplexP<dim>(params.finite_element.degree));
      quadrature = std::make_shared<dealii::QGaussSimplex<dim>>(params.finite_element.degree + 1);
      dof_handler.distribute_dofs(dealii::FE_SimplexP<dim>(params.finite_element.degree));
    }
  else
    {
      quadrature    = std::make_shared<dealii::QGauss<dim>>(params.finite_element.degree + 1);
      const auto fe = dealii::FE_Q<dim>(params.finite_element.degree);
      dof_handler.distribute_dofs(fe);
      mapping = std::make_shared<dealii::MappingQ<dim>>(params.finite_element.degree);
    }

  print_entry(pcout, "Number of Finite Elements", triangulation->n_cells());
  print_entry(pcout, "Number of DoFs", dof_handler.n_dofs());

  if (params.multigrid.enable)
    dof_handler.distribute_mg_dofs();

  if (params.output.print_memory_usage)
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
  if (not params.problem.use_neumann_bc)
    {
      active_constraints.reinit(dof_handler.locally_owned_dofs(),
                                DoFTools::extract_locally_relevant_dofs(dof_handler));
      VectorTools::interpolate_boundary_values(
        *mapping, dof_handler, 0, *dbc_func, active_constraints);
    }
  active_constraints.close();

  MatrixType active_operator(params.problem.screening_length);
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
          const auto source_data_file = input_file_path.parent_path() / params.source_data.file;

          switch (params.source_data.assembly_strategy())
            {
                case SourceRhsAssemblyStrategy::source_point_quadrature: {
                  const auto output_file_name = params.output.name + "_source_points.vtu";

                  Otter::assemble_rhs_from_source_point_quadrature<dim, Number, VectorType>(
                    src,
                    *mapping,
                    *triangulation,
                    source_data_file.string(),
                    active_operator.get_matrix_free(),
                    output_file_name,
                    params.output.write_source_data,
                    params.output.print_memory_usage);

                  break;
                }

                case SourceRhsAssemblyStrategy::standard_quadrature: {
                  const auto output_file_name = params.output.name + "_source_points.vtk";

                  Otter::assemble_rhs_from_standard_quadrature<dim, Number, Number, VectorType>(
                    src,
                    *mapping,
                    source_data_file.string(),
                    active_operator.get_matrix_free(),
                    output_file_name,
                    params.output.write_source_data,
                    params.output.print_memory_usage);

                  break;
                }

                case SourceRhsAssemblyStrategy::standard_quadrature_fast: {
                  const auto output_file_name = params.output.name + "_source_points.vtk";

                  Otter::assemble_rhs_from_standard_quadrature_fast<dim, Number, VectorType>(
                    src,
                    *mapping,
                    source_data_file.string(),
                    active_operator.get_matrix_free(),
                    output_file_name,
                    params.source_data.is_regular_grid,
                    params.output.write_source_data,
                    params.output.print_memory_usage);

                  break;
                }
            }
        }

      print_entry(pcout, "||RHS||l2", src.l2_norm());
      timer.leave_subsection();
    }

  if (params.output.print_memory_usage)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "create_rhs");
    }

  // Finally, solve.
  ReductionControl solver_control(params.linear_solver.max_iterations,
                                  params.linear_solver.absolute_tolerance,
                                  params.linear_solver.relative_tolerance,
                                  false,
                                  false);

  if (params.multigrid.enable)
    {
      timer.enter_subsection("Setup GMG");
      print_entry(pcout, "Preconditioner", "GMG");
      MGLevelObject<AffineConstraints<NumberMG>> mg_constraints(min_level, max_level);
      MGLevelObject<LevelMatrixType>             mg_operators(min_level,
                                                  max_level,
                                                  params.problem.screening_length);

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
          smoother_data[level].smoothing_range     = params.multigrid.smoother.smoothing_range;
          smoother_data[level].degree              = params.multigrid.smoother.degree;
          smoother_data[level].eig_cg_n_iterations = params.multigrid.smoother.eigen_cg_iterations;
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
      ReductionControl coarse_grid_solver_control(params.multigrid.coarse_solver.max_iterations,
                                                  params.multigrid.coarse_solver.absolute_tolerance,
                                                  params.multigrid.coarse_solver.relative_tolerance,
                                                  false,
                                                  false);
      SolverGMRES<VectorTypeDouble> coarse_grid_solver(coarse_grid_solver_control);

      PreconditionIdentity precondition_identity;
      PreconditionChebyshev<LevelMatrixType, VectorTypeMG, DiagonalMatrix<VectorTypeMG>>
        precondition_chebyshev;

      TrilinosWrappers::PreconditionAMG precondition_amg;
      TrilinosWrappers::SolverDirect    precondition_direct;

      std::unique_ptr<MGCoarseGridBase<VectorTypeMG>> mg_coarse;

      if (params.multigrid.coarse_solver.type == "gmres_with_amg")
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          amg_data.smoother_sweeps = params.multigrid.coarse_solver.smoother_sweeps;
          amg_data.n_cycles        = params.multigrid.coarse_solver.n_v_cycles;
          amg_data.smoother_type   = params.multigrid.coarse_solver.smoother_type.c_str();

          // GMRES with AMG as preconditioner
          precondition_amg.initialize(mg_operators[min_level].get_system_matrix(), amg_data);

          mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<VectorTypeMG,
                                                                   SolverGMRES<VectorTypeDouble>,
                                                                   TrilinosWrappers::SparseMatrix,
                                                                   decltype(precondition_amg)>>(
            coarse_grid_solver, mg_operators[min_level].get_system_matrix(), precondition_amg);
        }
      else if (params.multigrid.coarse_solver.type == "direct")
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

  if (params.output.print_memory_usage)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_after, mem_after_solve, MPI_COMM_WORLD, "solve");
    }

  if (params.output.compute_solution_l2)
    {
      // use a higher order quadrature rule for computing the L2 norm
      std::shared_ptr<Quadrature<dim>> quadrature_L2;
      if (params.mesh.use_simplex_mesh)
        quadrature_L2 = std::make_shared<dealii::QGaussSimplex<dim>>(
          std::min(static_cast<int>(params.finite_element.degree + 2), 4));
      else
        quadrature_L2 = std::make_shared<dealii::QGauss<dim>>(params.finite_element.degree + 2);

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
      print_section(pcout, "Post-Processing");

      print_entry(pcout, "||u||L2", error);
    }

  if (params.output.write_paraview)
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
      data_out.build_patches(*mapping, params.finite_element.degree);
      data_out.write_vtu_in_parallel(params.output.name + ".vtu", MPI_COMM_WORLD);
      timer.leave_subsection();
    }

  if (params.output.print_wall_times)
    timer.print_wall_time_statistics(MPI_COMM_WORLD);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Utilities::System::MemoryStats mem_before, mem_after;
  dealii::Utilities::System::get_memory_stats(mem_before);
  ScreenedPoissonParameters params;

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

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && params.output.verbosity > 0)
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

  if (params.problem.dimension == 1)
    run<1, double, float>(params, input_file_path);
  else if (params.problem.dimension == 2)
    run<2, double, float>(params, input_file_path);
  else if (params.problem.dimension == 3)
    run<3, double, float>(params, input_file_path);
  else
    {
      AssertThrow(false, ExcMessage("Your run command is wrong."));
    }

  if (params.output.print_memory_usage)
    {
      dealii::Utilities::System::get_memory_stats(mem_after);
      print_memory_stats(mem_before, mem_after, MPI_COMM_WORLD, "main");
    }
}
