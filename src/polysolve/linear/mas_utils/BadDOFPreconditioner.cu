#include <polysolve/linear/mas_utils/BadDOFPreconditioner.hpp>

#include <cuda/algorithm>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

namespace polysolve::linear::mas
{
    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }

        template <typename T>
        std::vector<T> copy_device_span_to_host(ctd::span<const T> src, CudaRuntime rt)
        {
            std::vector<T> out(src.size());
            if (src.empty())
            {
                return out;
            }

            cudaMemcpyAsync(
                out.data(),
                src.data(),
                src.size() * sizeof(T),
                cudaMemcpyDeviceToHost,
                rt.stream.get());
            rt.stream.sync();
            return out;
        }

        // true if not BSR padding.
        __both__ bool is_padding(BSRView view, int scalar_id)
        {
            int block = scalar_id / view.block_dim;
            int local = scalar_id % view.block_dim;
            return (block == view.padded_block
                    && local >= view.block_dim - view.padded_scalar_num);
        }

        struct HostBadDofMatrix
        {
            std::vector<int> indices;
            std::vector<int> rows;
            std::vector<int> cols;
            std::vector<double> vals;
        };

        std::vector<int> select_bad_dofs_from_diagonal(const std::vector<double> &diag_abs)
        {
            int n = diag_abs.size();
            if (n == 0)
            {
                return {};
            }

            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
                return diag_abs[lhs] < diag_abs[rhs];
            });

            int cutoff_index = 0;
            if (n > 1)
            {
                std::vector<double> log_sorted(n, 0.0);
                for (int i = 0; i < n; ++i)
                {
                    log_sorted[i] = std::log(diag_abs[order[i]]);
                }

                double min_mag = log_sorted.front();
                double max_mag = log_sorted.back();
                double max_deviation = -1.0;
                for (int i = 0; i < n; ++i)
                {
                    double expected_y = (max_mag - min_mag) / (n - 1.0) * i + min_mag;
                    double deviation = std::abs(expected_y - log_sorted[i]);
                    if (deviation > max_deviation)
                    {
                        max_deviation = deviation;
                        cutoff_index = i;
                    }
                }
            }

            return std::vector<int>(order.begin() + cutoff_index, order.end());
        }

        HostBadDofMatrix build_bad_dof_matrix(BSRView view, CudaRuntime rt)
        {
            HostBadDofMatrix out;
            auto h_rows = copy_device_span_to_host(view.rows, rt);
            auto h_cols = copy_device_span_to_host(view.cols, rt);
            auto h_vals = copy_device_span_to_host(view.vals, rt);

            // Gather permuted abs(diagonal) excluding BSR padding.
            int block_size = view.block_dim * view.block_dim;
            int permuted_scalar_dim = view.dim * view.block_dim;
            std::vector<double> diag_abs;
            std::vector<int> diag_idx;
            for (int block_row = 0; block_row < view.dim; ++block_row)
            {
                // Traverse BSR, find diagonal.
                int diag_block = -1;
                for (int p = h_rows[block_row]; p < h_rows[block_row + 1]; ++p)
                {
                    if (h_cols[p] == block_row)
                    {
                        diag_block = p;
                        break;
                    }
                }
                if (diag_block < 0)
                {
                    continue;
                }

                // Copy diagonals within block.
                int diag_offset = diag_block * block_size;
                for (int local_row = 0; local_row < view.block_dim; ++local_row)
                {
                    int scalar_row = block_row * view.block_dim + local_row;
                    if (is_padding(view, scalar_row))
                    {
                        continue;
                    }

                    double val = std::abs(h_vals[diag_offset + local_row * view.block_dim + local_row]);
                    diag_idx.push_back(scalar_row);
                    diag_abs.push_back(val);
                }
            }

            // Select bad dof and build gather/scatter map.
            std::vector<int> selected_positions = select_bad_dofs_from_diagonal(diag_abs);
            for (int selected_pos : selected_positions)
            {
                out.indices.push_back(diag_idx[selected_pos]);
            }

            std::vector<int> scalar_to_bad(permuted_scalar_dim, -1);
            for (int i = 0; i < out.indices.size(); ++i)
            {
                scalar_to_bad[out.indices[i]] = i;
            }

            // Gather bad dof subdomain A in CSR format.
            out.rows.push_back(0);
            for (int bad_row = 0; bad_row < out.indices.size(); ++bad_row)
            {
                int scalar_row = out.indices[bad_row];
                int block_row = scalar_row / view.block_dim;
                int local_row = scalar_row % view.block_dim;

                std::vector<std::pair<int, double>> row_entries;
                for (int p = h_rows[block_row]; p < h_rows[block_row + 1]; ++p)
                {
                    int block_col = h_cols[p];
                    int row_offset = p * block_size + local_row * view.block_dim;
                    for (int local_col = 0; local_col < view.block_dim; ++local_col)
                    {
                        int scalar_col = block_col * view.block_dim + local_col;
                        int bad_col = scalar_to_bad[scalar_col];
                        if (bad_col < 0)
                        {
                            continue;
                        }

                        double value = h_vals[row_offset + local_col];
                        if (value == 0.0)
                        {
                            continue;
                        }

                        row_entries.emplace_back(bad_col, value);
                    }
                }

                std::sort(
                    row_entries.begin(),
                    row_entries.end(),
                    [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

                for (const auto &[bad_col, value] : row_entries)
                {
                    out.cols.push_back(bad_col);
                    out.vals.push_back(value);
                }
                out.rows.push_back(out.cols.size());
            }

            return out;
        }

        __global__ void gather_selected_scalars(
            ctd::span<const double> input,
            ctd::span<const int> indices,
            ctd::span<double> output)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= indices.size())
            {
                return;
            }

            output[tid] = input[indices[tid]];
        }

        __global__ void scatter_selected_scalars(
            ctd::span<const double> input,
            ctd::span<const int> indices,
            ctd::span<double> output)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= indices.size())
            {
                return;
            }

            output[indices[tid]] = input[tid];
        }
    } // namespace

    void BadDOFPreconditioner::reset()
    {
        vector_size_ = 0;
        bad_dof_count_ = 0;
        cudss_b_ = {};
        cudss_x_ = {};
        cudss_A_ = {};
        cudss_data_ = {};

        bad_dof_indices_.reset();
        bad_dof_rows_.reset();
        bad_dof_cols_.reset();
        bad_dof_vals_.reset();
        bad_dof_rhs_.reset();
        bad_dof_solution_.reset();
    }

    void BadDOFPreconditioner::factorize(
        const BSRMatrix &A,
        CudaRuntime rt)
    {
        auto phase_begin = clock::now();
        reset();
        cudssSetStream(cudss_handle_.raw, rt.stream.get());

        BSRView view = A.view();
        vector_size_ = view.dim * view.block_dim;

        HostBadDofMatrix host_bad_dof = build_bad_dof_matrix(view, rt);
        bad_dof_count_ = host_bad_dof.indices.size();
        SPDLOG_INFO(
            "[MAS] [factorize_bad_dof] [{:.6f}] [count={}]",
            elapsed_seconds(phase_begin),
            bad_dof_count_);
        if (bad_dof_count_ == 0)
        {
            return;
        }

        bad_dof_indices_ = safe_alloc<int>(host_bad_dof.indices.size(), rt, "BadDOF factorize indices");
        bad_dof_rows_ = safe_alloc<int>(host_bad_dof.rows.size(), rt, "BadDOF factorize rows");
        bad_dof_cols_ = safe_alloc<int>(host_bad_dof.cols.size(), rt, "BadDOF factorize cols");
        bad_dof_vals_ = safe_alloc<double>(host_bad_dof.vals.size(), rt, "BadDOF factorize vals");
        bad_dof_rhs_ = safe_alloc<double>(bad_dof_count_, 0.0, rt, "BadDOF factorize rhs");
        bad_dof_solution_ = safe_alloc<double>(bad_dof_count_, 0.0, rt, "BadDOF factorize solution");

        cu::copy_bytes(rt.stream, host_bad_dof.indices, *bad_dof_indices_);
        cu::copy_bytes(rt.stream, host_bad_dof.rows, *bad_dof_rows_);
        cu::copy_bytes(rt.stream, host_bad_dof.cols, *bad_dof_cols_);
        cu::copy_bytes(rt.stream, host_bad_dof.vals, *bad_dof_vals_);
        rt.stream.sync();

        cudss_data_ = CuDSSData(cudss_handle_.raw);
        cudss_A_ = CuDSSMatrix(
            bad_dof_count_,
            bad_dof_count_,
            host_bad_dof.vals.size(),
            bad_dof_rows_->data(),
            bad_dof_cols_->data(),
            bad_dof_vals_->data());
        cudss_x_ = CuDSSMatrix(
            bad_dof_count_,
            1,
            bad_dof_count_,
            bad_dof_solution_->data());
        cudss_b_ = CuDSSMatrix(
            bad_dof_count_,
            1,
            bad_dof_count_,
            bad_dof_rhs_->data());

        cudssExecute(
            cudss_handle_.raw,
            CUDSS_PHASE_REORDERING,
            cudss_config_.raw,
            cudss_data_.raw,
            cudss_A_.raw,
            nullptr,
            nullptr);
        cudssExecute(
            cudss_handle_.raw,
            CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
            cudss_config_.raw,
            cudss_data_.raw,
            cudss_A_.raw,
            nullptr,
            nullptr);
        cudssExecute(
            cudss_handle_.raw,
            CUDSS_PHASE_FACTORIZATION,
            cudss_config_.raw,
            cudss_data_.raw,
            cudss_A_.raw,
            nullptr,
            nullptr);
        rt.stream.sync();
    }

    void BadDOFPreconditioner::apply(
        ctd::span<const double> r,
        ctd::span<double> z,
        CudaRuntime rt)
    {
        cudssSetStream(cudss_handle_.raw, rt.stream.get());
        cu::fill_bytes(rt.stream, z, 0);
        if (bad_dof_count_ == 0)
        {
            return;
        }

        cu::fill_bytes(rt.stream, *(bad_dof_rhs_), 0);
        cu::fill_bytes(rt.stream, *(bad_dof_solution_), 0);

        int bad_grid_num = div_round_up(bad_dof_count_, 128);
        gather_selected_scalars<<<bad_grid_num, 128, 0, rt.stream.get()>>>(
            r,
            *(bad_dof_indices_),
            *(bad_dof_rhs_));

        rt.stream.sync();
        cudssExecute(
            cudss_handle_.raw,
            CUDSS_PHASE_SOLVE,
            cudss_config_.raw,
            cudss_data_.raw,
            cudss_A_.raw,
            cudss_x_.raw,
            cudss_b_.raw);
        rt.stream.sync();

        scatter_selected_scalars<<<bad_grid_num, 128, 0, rt.stream.get()>>>(
            *(bad_dof_solution_),
            *(bad_dof_indices_),
            z);
    }
} // namespace polysolve::linear::mas
