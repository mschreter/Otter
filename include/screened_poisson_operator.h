#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.templates.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_tools.h>

using namespace dealii;

template <int dim, typename Number, typename VectorizedArrayType = VectorizedArray<Number>>
class ScreenedPoissonOperator : public Subscriptor
{
public:
  using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  ScreenedPoissonOperator(const double screening_length)
    : screening_length_sq(screening_length * screening_length)
  {}

  ScreenedPoissonOperator(const Mapping<dim>              &mapping,
                          const DoFHandler<dim>           &dof_handler,
                          const AffineConstraints<Number> &constraints,
                          const Quadrature<dim>           &quadrature,
                          const double                     screening_length)
    : screening_length_sq(screening_length * screening_length)
  {
    reinit(mapping, dof_handler, constraints, quadrature);
  }

  void
  reinit(const Mapping<dim>              &mapping,
         const DoFHandler<dim>           &dof_handler,
         const AffineConstraints<Number> &constraints,
         const Quadrature<dim>           &quadrature,
         const unsigned int               mg_level = numbers::invalid_unsigned_int)
  {
    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData data;
    data.mapping_update_flags = update_quadrature_points | update_gradients | update_values;
    data.mg_level             = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quadrature, data);

    valid_system = false;
  }

  virtual types::global_dof_index
  m() const
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    DEAL_II_NOT_IMPLEMENTED();
    return 0;
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void
  initialize_dof_vector(VectorType &dst) const
  {
    matrix_free.initialize_dof_vector(dst);
  }

  template <typename FunctionType>
  void
  rhs(VectorType &system_rhs, const FunctionType &rhs_func) const
  {
    const int dummy = 0;

    if (rhs_func)
      {
        matrix_free.template cell_loop<VectorType, int>(
          [&](const auto &data, auto &dst, const auto &, const auto cells) {
            FECellIntegrator phi(data, 0 /*dof_idx*/, 0 /*quad_idx*/);
            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
              {
                phi.reinit(cell);
                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  {
                    VectorizedArrayType coeff = 0;

                    const auto point_batch = phi.quadrature_point(q);

                    for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                      {
                        Point<dim> single_point;
                        for (unsigned int d = 0; d < dim; d++)
                          single_point[d] = point_batch[d][v];
                        coeff[v] = rhs_func->value(single_point);
                      }

                    phi.submit_value(coeff, q);
                  }

                phi.integrate_scatter(EvaluationFlags::values, dst);
              }
          },
          system_rhs,
          dummy,
          true);
      }
    else
      {
        system_rhs = 0.0;
      }

    VectorType b, x;

    this->initialize_dof_vector(b);
    this->initialize_dof_vector(x);

    typename MatrixFree<dim, Number>::AdditionalData data;
    data.mapping_update_flags = update_values | update_gradients | update_quadrature_points;

    MatrixFree<dim, Number> matrix_free;
    matrix_free.reinit(*this->matrix_free.get_mapping_info().mapping,
                       this->matrix_free.get_dof_handler(),
                       AffineConstraints<Number>(),
                       this->matrix_free.get_quadrature(),
                       data);

    // set constrained
    this->matrix_free.get_affine_constraints().distribute(x);

    // perform matrix-vector multiplication (with unconstrained system and
    // constrained set in vector)
    matrix_free.cell_loop(
      &ScreenedPoissonOperator<dim, Number, VectorizedArrayType>::do_vmult_cell, this, b, x, true);

    // clear constrained values
    this->matrix_free.get_affine_constraints().set_zero(b);

    // move to the right-hand side
    system_rhs -= b;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.cell_loop(&ScreenedPoissonOperator<dim, Number, VectorizedArrayType>::do_vmult_cell,
                          this,
                          dst,
                          src,
                          true);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    matrix_free.initialize_dof_vector(diagonal);

    MatrixFreeTools::compute_diagonal(
      matrix_free,
      diagonal,
      &ScreenedPoissonOperator<dim, Number, VectorizedArrayType>::do_vmult_cell_single,
      this);

    for (auto &i : diagonal)
      i = (i != 0.0) ? (1.0 / i) : 1.0;
  }

  const TrilinosWrappers::SparseMatrix &
  get_system_matrix() const
  {
    initialize_system_matrix();

    return system_matrix;
  }

  void
  initialize_system_matrix() const
  {
    const auto &dof_handler = matrix_free.get_dof_handler();

    const auto &constraints = matrix_free.get_affine_constraints();

    if (system_matrix.m() == 0 || system_matrix.n() == 0)
      {
        system_matrix.clear();

        TrilinosWrappers::SparsityPattern dsp(
          this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_mg_dofs(this->matrix_free.get_mg_level()) :
            dof_handler.locally_owned_dofs(),
          matrix_free.get_task_info().communicator);

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_sparsity_pattern(dof_handler,
                                         dsp,
                                         this->matrix_free.get_mg_level(),
                                         constraints);
        else
          DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

        dsp.compress();

        system_matrix.reinit(dsp);
      }

    if (this->valid_system == false)
      {
        system_matrix = 0.0;

        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          system_matrix,
          &ScreenedPoissonOperator<dim, Number, VectorizedArrayType>::do_vmult_cell_single,
          this);

        system_matrix.compress(VectorOperation::add);

        this->valid_system = true;
      }
  }

const VectorType&
get_diagonal() const
{
  initialize_diagonal();
  return diagonal;
}

void
initialize_diagonal() const
{
  diagonal.reinit(matrix_free.get_vector_partitioner());

  MatrixFreeTools::compute_diagonal(
    matrix_free,
    diagonal,
    &ScreenedPoissonOperator<dim,
                             Number,
                             VectorizedArrayType>::do_vmult_cell_single,
    this);

  diagonal.compress(VectorOperation::insert);
}

  const MatrixFree<dim, Number, VectorizedArrayType> &
  get_matrix_free()
  {
    return matrix_free;
  }

private:
  const double                                 screening_length_sq = 0;
  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

  void
  do_vmult_cell(const MatrixFree<dim, Number>               &data,
                VectorType                                  &dst,
                const VectorType                            &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        do_vmult_cell_single(phi);

        phi.distribute_local_to_global(dst);
      }
  }

  void
  do_vmult_cell_single(FECellIntegrator &phi) const
  {
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        phi.submit_value(phi.get_value(q), q);
        phi.submit_gradient(screening_length_sq * phi.get_gradient(q), q);
      }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  mutable TrilinosWrappers::SparseMatrix system_matrix;
  mutable VectorType                     diagonal;
  mutable bool                           valid_system;
};
