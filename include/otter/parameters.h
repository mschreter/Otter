#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <string>

namespace Otter
{
  /**
   * @brief Available strategies for assembling the screened-Poisson
   * right-hand side from source data.
   */
  enum class SourceRhsAssemblyStrategy
  {
    source_point_quadrature,
    standard_quadrature,
    standard_quadrature_fast
  };



  inline std::string
  to_string(const SourceRhsAssemblyStrategy strategy)
  {
    switch (strategy)
      {
        case SourceRhsAssemblyStrategy::source_point_quadrature:
          return "source_point_quadrature";

        case SourceRhsAssemblyStrategy::standard_quadrature:
          return "standard_quadrature";

        case SourceRhsAssemblyStrategy::standard_quadrature_fast:
          return "standard_quadrature_fast";
      }

    AssertThrow(false, dealii::ExcNotImplemented());
    return "";
  }



  inline SourceRhsAssemblyStrategy
  source_rhs_assembly_strategy_from_string(const std::string &name)
  {
    if (name == "source_point_quadrature")
      return SourceRhsAssemblyStrategy::source_point_quadrature;

    if (name == "standard_quadrature")
      return SourceRhsAssemblyStrategy::standard_quadrature;

    if (name == "standard_quadrature_fast")
      return SourceRhsAssemblyStrategy::standard_quadrature_fast;

    AssertThrow(false,
                dealii::ExcMessage("Unknown source RHS assembly strategy: " + name));

    return SourceRhsAssemblyStrategy::standard_quadrature;
  }



  struct ProblemParameters
  {
    int    dimension        = 3;
    double screening_length = 0.0;
    bool   use_neumann_bc   = true;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("problem");
      {
        prm.add_parameter("dimension", dimension, "Spatial dimension.");
        prm.add_parameter("screening length",
                          screening_length,
                          "Screening length of the screened-Poisson equation.");
        prm.add_parameter("use neumann boundary conditions",
                          use_neumann_bc,
                          "Use homogeneous Neumann boundary conditions. "
                          "Alternatively, Dirichlet boundary conditions with "
                          "a prescribed value of 1 are used.");
      }
      prm.leave_subsection();
    }
  };



  struct MeshParameters
  {
    std::string  type                 = "";
    std::string  arguments            = "";
    unsigned int n_global_refinements = 5;
    bool         use_simplex_mesh     = false;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("mesh");
      {
        prm.add_parameter("type",
                          type,
                          "Mesh type or mesh file name. Examples: hyper_cube, "
                          "hyper_rectangle, cylinder, or a mesh file.");
        prm.add_parameter("arguments",
                          arguments,
                          "Arguments passed to "
                          "GridGenerator::generate_from_name_and_arguments().");
        prm.add_parameter("global refinements",
                          n_global_refinements,
                          "Number of global mesh refinements.");
        prm.add_parameter("use simplex mesh",
                          use_simplex_mesh,
                          "Use simplex elements instead of tensor-product elements.");
      }
      prm.leave_subsection();
    }
  };



  struct FiniteElementParameters
  {
    unsigned int degree = 1;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("finite element");
      {
        prm.add_parameter("degree",
                          degree,
                          "Polynomial degree of the finite element space.");
      }
      prm.leave_subsection();
    }
  };



  struct SourceDataParameters
  {
    std::string file            = "";
    bool        is_regular_grid = false;

    std::string analytical_function = "";

    std::string rhs_assembly_strategy =
      to_string(SourceRhsAssemblyStrategy::standard_quadrature);

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("source data");
      {
        prm.add_parameter("file",
                          file,
                          "Source-data file. Supported formats depend on the selected "
                          "RHS assembly strategy.");
        prm.add_parameter("is regular grid",
                          is_regular_grid,
                          "Specify whether the source data is stored on a regular grid "
                          "with spacing 1.");
        prm.add_parameter("analytical function",
                          analytical_function,
                          "Analytical right-hand-side function. If this is non-empty, "
                          "source-data assembly is skipped.");
        prm.add_parameter("rhs assembly strategy",
                          rhs_assembly_strategy,
                          "Choose how source data is integrated into the finite-element "
                          "right-hand side.",
                          dealii::Patterns::Selection(
                            "source_point_quadrature|standard_quadrature|standard_quadrature_fast"));
      }
      prm.leave_subsection();
    }

    SourceRhsAssemblyStrategy
    assembly_strategy() const
    {
      return source_rhs_assembly_strategy_from_string(rhs_assembly_strategy);
    }
  };



  struct LinearSolverParameters
  {
    unsigned int max_iterations     = 10000;
    double       absolute_tolerance = 1e-20;
    double       relative_tolerance = 1e-10;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("linear solver");
      {
        prm.add_parameter("max iterations",
                          max_iterations,
                          "Maximum number of linear solver iterations.");
        prm.add_parameter("absolute tolerance",
                          absolute_tolerance,
                          "Absolute convergence tolerance.");
        prm.add_parameter("relative tolerance",
                          relative_tolerance,
                          "Relative convergence tolerance.");
      }
      prm.leave_subsection();
    }
  };



  struct MultigridSmootherParameters
  {
    std::string  type                = "chebyshev";
    double       smoothing_range     = 20.0;
    unsigned int degree              = 5;
    unsigned int eigen_cg_iterations = 20;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("smoother");
      {
        prm.add_parameter("type",
                          type,
                          "Type of multigrid smoother, e.g., chebyshev.");
        prm.add_parameter("smoothing range",
                          smoothing_range,
                          "Smoothing range used by the smoother.");
        prm.add_parameter("degree",
                          degree,
                          "Degree of the smoother polynomial.");
        prm.add_parameter("eigenvalue cg iterations",
                          eigen_cg_iterations,
                          "Number of CG iterations for estimating the largest eigenvalue.");
      }
      prm.leave_subsection();
    }
  };



  struct MultigridCoarseSolverParameters
  {
    std::string  type               = "direct";
    unsigned int max_iterations     = 10000;
    double       absolute_tolerance = 1e-20;
    double       relative_tolerance = 1e-4;

    unsigned int smoother_sweeps = 1;
    unsigned int n_v_cycles      = 1;
    std::string  smoother_type   = "ILU";

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("coarse solver");
      {
        prm.add_parameter("type",
                          type,
                          "Type of coarse-grid solver.");
        prm.add_parameter("max iterations",
                          max_iterations,
                          "Maximum number of coarse-grid solver iterations.");
        prm.add_parameter("absolute tolerance",
                          absolute_tolerance,
                          "Absolute coarse-grid solver tolerance.");
        prm.add_parameter("relative tolerance",
                          relative_tolerance,
                          "Relative coarse-grid solver tolerance.");
        prm.add_parameter("smoother sweeps",
                          smoother_sweeps,
                          "Number of coarse-grid smoother sweeps.");
        prm.add_parameter("v-cycle steps",
                          n_v_cycles,
                          "Number of coarse-grid V-cycle steps.");
        prm.add_parameter("smoother type",
                          smoother_type,
                          "Coarse-grid smoother type, e.g., ILU.");
      }
      prm.leave_subsection();
    }
  };



  struct MultigridParameters
  {
    bool enable = false;

    MultigridSmootherParameters     smoother;
    MultigridCoarseSolverParameters coarse_solver;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("geometric multigrid");
      {
        prm.add_parameter("enable",
                          enable,
                          "Enable geometric multigrid. If false, AMG is used instead.");

        smoother.add_parameters(prm);
        coarse_solver.add_parameters(prm);
      }
      prm.leave_subsection();
    }
  };



  struct OutputParameters
  {
    std::string  name                  = "output";
    bool         write_paraview        = true;
    bool         write_source_data     = true;
    bool         compute_solution_l2   = true;
    bool         print_wall_times      = false;
    bool         print_memory_usage    = false;
    unsigned int verbosity             = 0;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("output");
      {
        prm.add_parameter("name",
                          name,
                          "Base name of the output files.");
        prm.add_parameter("write paraview files",
                          write_paraview,
                          "Write solution output for ParaView.");
        prm.add_parameter("write source data",
                          write_source_data,
                          "Write source-data output.");
        prm.add_parameter("compute solution l2 norm",
                          compute_solution_l2,
                          "Compute the L2 norm of the solution.");
        prm.add_parameter("print wall times",
                          print_wall_times,
                          "Print wall-time statistics.");
        prm.add_parameter("print memory usage",
                          print_memory_usage,
                          "Print memory usage statistics.");
        prm.add_parameter("verbosity",
                          verbosity,
                          "Verbosity level. Use 0 for tests and 1 for detailed output.");
      }
      prm.leave_subsection();
    }
  };



  struct ScreenedPoissonParameters
  {
    ProblemParameters       problem;
    MeshParameters          mesh;
    FiniteElementParameters finite_element;
    SourceDataParameters    source_data;
    LinearSolverParameters  linear_solver;
    MultigridParameters     multigrid;
    OutputParameters        output;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      problem.add_parameters(prm);
      mesh.add_parameters(prm);
      finite_element.add_parameters(prm);
      source_data.add_parameters(prm);
      linear_solver.add_parameters(prm);
      multigrid.add_parameters(prm);
      output.add_parameters(prm);
    }
  };
} // namespace Otter
