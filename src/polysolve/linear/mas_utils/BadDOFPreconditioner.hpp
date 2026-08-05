#pragma once

#include <polysolve/linear/mas_utils/BSRMatrix.hpp>
#include <polysolve/linear/mas_utils/CuDSSWrapper.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    class BadDOFPreconditioner
    {
    public:
        bool empty() const
        {
            return vector_size_ == 0;
        }

        void factorize(const BSRMatrix &A, CudaRuntime rt);
        void apply(ctd::span<const double> r, ctd::span<double> z, CudaRuntime rt);

    private:
        int vector_size_ = 0;
        int bad_dof_count_ = 0;

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
