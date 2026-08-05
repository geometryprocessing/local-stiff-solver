#pragma once

#include <polysolve/linear/mas_utils/BSRMatrix.hpp>
#include <polysolve/linear/mas_utils/CuDSSWrapper.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    class BadDofGMMPreconditioner
    {
    public:
        bool empty() const
        {
            return vector_size_ == 0;
        }

        void set_gmm_jump_threshold(double jump_threshold)
        {
            gmm_jump_threshold_ = jump_threshold;
        }

        void set_gmm_tol(double tol)
        {
            gmm_tol_ = tol;
        }

        void set_gmm_max_iterations(int max_iterations)
        {
            gmm_max_iterations_ = max_iterations;
        }

        void set_gmm_expand_neighbors(int expand_neighbors)
        {
            gmm_expand_neighbors_ = (expand_neighbors > 0) ? expand_neighbors : 0;
        }

        void factorize(const BSRMatrix &A, CudaRuntime rt);
        void apply(ctd::span<const double> r, ctd::span<double> z, CudaRuntime rt);

    private:
        int vector_size_ = 0;
        int bad_dof_count_ = 0;
        double gmm_jump_threshold_ = 10.0;
        double gmm_tol_ = 1e-3;
        int gmm_max_iterations_ = 5;
        int gmm_expand_neighbors_ = 32;

        Buf<int> bad_dof_indices_;
        Buf<int> bad_dof_rows_;
        Buf<int> bad_dof_cols_;
        Buf<double> bad_dof_vals_;
        Buf<double> bad_dof_rhs_;
        Buf<double> bad_dof_solution_;

        CuDSSHandle cudss_handle_;
        CuDSSConfig cudss_config_;
        CuDSSData cudss_data_;
        CuDSSMatrix cudss_A_;
        CuDSSMatrix cudss_x_;
        CuDSSMatrix cudss_b_;

        void reset();
    };
} // namespace polysolve::linear::mas
