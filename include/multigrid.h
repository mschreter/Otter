#pragma once

namespace dealii
{
  /**
   * Coarse grid solver using a preconditioner only. This is a little wrapper,
   * transforming a preconditioner into a coarse grid solver.
   */
  template <class VectorType, class PreconditionerType>
  class MGCoarseGridApplyPreconditioner : public MGCoarseGridBase<VectorType>
  {
  public:
    /**
     * Default constructor.
     */
    MGCoarseGridApplyPreconditioner();

    /**
     * Constructor. Store a pointer to the preconditioner for later use.
     */
    MGCoarseGridApplyPreconditioner(const PreconditionerType &precondition);

    /**
     * Clear the pointer.
     */
    void
    clear();

    /**
     * Initialize new data.
     */
    void
    initialize(const PreconditionerType &precondition);

    /**
     * Implementation of the abstract function.
     */
    virtual void
    operator()(const unsigned int level, VectorType &dst, const VectorType &src) const override;

  private:
    /**
     * Reference to the preconditioner.
     */
    ObserverPointer<const PreconditionerType,
                    MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>>
      preconditioner;
  };



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::MGCoarseGridApplyPreconditioner()
    : preconditioner(0, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::MGCoarseGridApplyPreconditioner(
    const PreconditionerType &preconditioner)
    : preconditioner(&preconditioner, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::initialize(
    const PreconditionerType &preconditioner_)
  {
    preconditioner = &preconditioner_;
  }



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::clear()
  {
    preconditioner = 0;
  }



  namespace internal
  {
    namespace MGCoarseGridApplyPreconditioner
    {
      template <
        class VectorType,
        class PreconditionerType,
        typename std::enable_if<std::is_same<typename VectorType::value_type, double>::value,
                                VectorType>::type * = nullptr>
      void
      solve(const PreconditionerType preconditioner, VectorType &dst, const VectorType &src)
      {
        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst, src);
      }

      template <
        class VectorType,
        class PreconditionerType,
        typename std::enable_if<!std::is_same<typename VectorType::value_type, double>::value,
                                VectorType>::type * = nullptr>
      void
      solve(const PreconditionerType preconditioner, VectorType &dst, const VectorType &src)
      {
        LinearAlgebra::distributed::Vector<double> src_;
        LinearAlgebra::distributed::Vector<double> dst_;

        src_ = src;
        dst_ = dst;

        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst_, src_);

        dst = dst_;
      }
    } // namespace MGCoarseGridApplyPreconditioner
  }   // namespace internal


  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::operator()(
    const unsigned int /*level*/,
    VectorType       &dst,
    const VectorType &src) const
  {
    internal::MGCoarseGridApplyPreconditioner::solve(preconditioner, dst, src);
  }
} // namespace dealii
