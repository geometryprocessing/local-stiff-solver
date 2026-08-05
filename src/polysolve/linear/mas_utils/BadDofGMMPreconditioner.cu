#include <polysolve/linear/mas_utils/BadDofGMMPreconditioner.hpp>

#include <cuda/algorithm>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
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

        // true if BSR padding.
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

        struct GMMSelectionStats
        {
            double global_mean = 0.0;
            double global_var = 0.0;
            double mean_0 = 0.0;
            double mean_1 = 0.0;
            double var_0 = 0.0;
            double var_1 = 0.0;
            int iterations = 0;
        };

        double relative_change(double lhs, double rhs)
        {
            constexpr double eps = 1e-300;
            return std::abs(lhs - rhs) / std::max(std::abs(rhs), eps);
        }

        double log_gaussian(double x, double mean, double var)
        {
            constexpr double pi = 3.141592653589793238462643383279502884;
            return -0.5 * std::log(2.0 * pi * var)
                   - 0.5 * (x - mean) * (x - mean) / var;
        }

        std::vector<int> select_bad_dofs_from_l1_row_norm_gmm(
            const std::vector<double> &row_norms,
            double jump_threshold,
            double tol,
            int max_iterations,
            int expand_neighbors,
            GMMSelectionStats &stats)
        {
            int n = row_norms.size();
            if (n < 2)
            {
                return {};
            }

            double sum_norms = 0.0;
            for (double norm : row_norms)
            {
                sum_norms += norm;
            }
            stats.global_mean = sum_norms / n;

            double var_sum = 0.0;
            for (double norm : row_norms)
            {
                double diff = norm - stats.global_mean;
                var_sum += diff * diff;
            }
            stats.global_var = var_sum / n;

            auto [min_it, max_it] = std::minmax_element(row_norms.begin(), row_norms.end());
            double mean_0 = *min_it;
            double mean_1 = *max_it;
            constexpr double var_reg = 1e-6;
            double var_0 = std::max(stats.global_var, var_reg);
            double var_1 = std::max(stats.global_var, var_reg);
            double w0 = 0.5;
            double w1 = 0.5;

            stats.mean_0 = mean_0;
            stats.mean_1 = mean_1;
            stats.var_0 = var_0;
            stats.var_1 = var_1;

            if (mean_0 == mean_1 || !std::isfinite(stats.global_var))
            {
                return {};
            }

            std::vector<double> gamma0(n, 0.0);
            std::vector<double> gamma1(n, 0.0);
            int iterations = std::max(1, max_iterations);
            int gmm_iter = 0;
            for (; gmm_iter < iterations; ++gmm_iter)
            {
                for (int i = 0; i < n; ++i)
                {
                    double x = row_norms[i];
                    double log_g0 = std::log(std::max(w0, std::numeric_limits<double>::min()))
                                    + log_gaussian(x, mean_0, var_0);
                    double log_g1 = std::log(std::max(w1, std::numeric_limits<double>::min()))
                                    + log_gaussian(x, mean_1, var_1);
                    double max_log_g = std::max(log_g0, log_g1);
                    double log_total = max_log_g
                                       + std::log(std::exp(log_g0 - max_log_g)
                                                  + std::exp(log_g1 - max_log_g));
                    gamma0[i] = std::exp(log_g0 - log_total);
                    gamma1[i] = std::exp(log_g1 - log_total);
                }

                double sum_g0 = 0.0;
                double sum_g1 = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    sum_g0 += gamma0[i];
                    sum_g1 += gamma1[i];
                }
                if (sum_g0 <= 0.0 || sum_g1 <= 0.0)
                {
                    break;
                }

                w0 = sum_g0 / n;
                w1 = sum_g1 / n;

                double old_mean_0 = mean_0;
                double old_mean_1 = mean_1;
                double old_var_0 = var_0;
                double old_var_1 = var_1;

                mean_0 = 0.0;
                mean_1 = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    mean_0 += gamma0[i] * row_norms[i];
                    mean_1 += gamma1[i] * row_norms[i];
                }
                mean_0 /= sum_g0;
                mean_1 /= sum_g1;

                var_0 = 0.0;
                var_1 = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    double diff_0 = row_norms[i] - mean_0;
                    double diff_1 = row_norms[i] - mean_1;
                    var_0 += gamma0[i] * diff_0 * diff_0;
                    var_1 += gamma1[i] * diff_1 * diff_1;
                }
                var_0 = var_0 / sum_g0 + var_reg;
                var_1 = var_1 / sum_g1 + var_reg;

                stats.mean_0 = mean_0;
                stats.mean_1 = mean_1;
                stats.var_0 = var_0;
                stats.var_1 = var_1;
                if (relative_change(mean_0, old_mean_0) < tol
                    && relative_change(mean_1, old_mean_1) < tol
                    && relative_change(var_0, old_var_0) < tol
                    && relative_change(var_1, old_var_1) < tol)
                {
                    break;
                }
            }
            stats.iterations = gmm_iter;

            int high_cluster = (std::abs(mean_0) > std::abs(mean_1)) ? 0 : 1;
            double high_mean = (high_cluster == 0) ? mean_0 : mean_1;
            double low_mean = (high_cluster == 0) ? mean_1 : mean_0;
            double ratio = std::abs(high_mean) / std::abs(low_mean);
            if (std::isnan(ratio) || ratio < jump_threshold)
            {
                return {};
            }

            if (expand_neighbors <= 0)
            {
                std::vector<int> selected_positions;
                for (int i = 0; i < n; ++i)
                {
                    if ((high_cluster == 0 && gamma0[i] > gamma1[i])
                        || (high_cluster == 1 && gamma1[i] > gamma0[i]))
                    {
                        selected_positions.push_back(i);
                    }
                }
                return selected_positions;
            }

            std::vector<unsigned char> selected_mask(n, 0);
            std::vector<int> selected_positions;
            for (int i = 0; i < n; ++i)
            {
                bool selected = (high_cluster == 0 && gamma0[i] > gamma1[i])
                                || (high_cluster == 1 && gamma1[i] > gamma0[i]);
                if (!selected)
                {
                    continue;
                }

                int l = std::max(0, i - expand_neighbors);
                int r = std::min(n - 1, i + expand_neighbors);
                for (int j = l; j <= r; ++j)
                {
                    if (selected_mask[j] == 0)
                    {
                        selected_mask[j] = 1;
                        selected_positions.push_back(j);
                    }
                }
            }

            return selected_positions;
        }

        HostBadDofMatrix build_bad_dof_matrix(
            BSRView view,
            double gmm_jump_threshold,
            double gmm_tol,
            int gmm_max_iterations,
            int gmm_expand_neighbors,
            GMMSelectionStats &stats,
            CudaRuntime rt)
        {
            HostBadDofMatrix out;
            auto h_rows = copy_device_span_to_host(view.rows, rt);
            auto h_cols = copy_device_span_to_host(view.cols, rt);
            auto h_vals = copy_device_span_to_host(view.vals, rt);

            // Gather permuted l1 row norm excluding BSR padding.
            int block_size = view.block_dim * view.block_dim;
            int permuted_scalar_dim = view.dim * view.block_dim;
            std::vector<double> row_l1_norms;
            std::vector<int> diag_idx;
            for (int block_row = 0; block_row < view.dim; ++block_row)
            {
                for (int local_row = 0; local_row < view.block_dim; ++local_row)
                {
                    int scalar_row = block_row * view.block_dim + local_row;
                    if (is_padding(view, scalar_row))
                    {
                        continue;
                    }

                    double val = 0.0;
                    for (int p = h_rows[block_row]; p < h_rows[block_row + 1]; ++p)
                    {
                        int block_col = h_cols[p];
                        int row_offset = p * block_size + local_row * view.block_dim;
                        for (int local_col = 0; local_col < view.block_dim; ++local_col)
                        {
                            int scalar_col = block_col * view.block_dim + local_col;
                            if (is_padding(view, scalar_col))
                            {
                                continue;
                            }

                            val += std::abs(h_vals[row_offset + local_col]);
                        }
                    }

                    diag_idx.push_back(scalar_row);
                    row_l1_norms.push_back(val);
                }
            }

            // Select bad dof and build gather/scatter map.
            std::vector<int> selected_positions = select_bad_dofs_from_l1_row_norm_gmm(
                row_l1_norms,
                gmm_jump_threshold,
                gmm_tol,
                gmm_max_iterations,
                gmm_expand_neighbors,
                stats);
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

    void BadDofGMMPreconditioner::reset()
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

    void BadDofGMMPreconditioner::factorize(
        const BSRMatrix &A,
        CudaRuntime rt)
    {
        auto phase_begin = clock::now();
        reset();
        cudssSetStream(cudss_handle_.raw, rt.stream.get());

        BSRView view = A.view();
        vector_size_ = view.dim * view.block_dim;

        GMMSelectionStats stats;
        HostBadDofMatrix host_bad_dof = build_bad_dof_matrix(
            view,
            gmm_jump_threshold_,
            gmm_tol_,
            gmm_max_iterations_,
            gmm_expand_neighbors_,
            stats,
            rt);
        bad_dof_count_ = host_bad_dof.indices.size();
        SPDLOG_INFO(
            "[MAS] [factorize_bad_dof_gmm] [{:.6f}] [global_mean={}] [global_var={}] [mean_0={}] [mean_1={}] [var_0={}] [var_1={}] [gmm_iters={}] [count={}]",
            elapsed_seconds(phase_begin),
            stats.global_mean,
            stats.global_var,
            stats.mean_0,
            stats.mean_1,
            stats.var_0,
            stats.var_1,
            stats.iterations,
            bad_dof_count_);
        if (bad_dof_count_ == 0)
        {
            return;
        }

        bad_dof_indices_ = safe_alloc<int>(host_bad_dof.indices.size(), rt, "BadDOF gmm factorize indices");
        bad_dof_rows_ = safe_alloc<int>(host_bad_dof.rows.size(), rt, "BadDOF gmm factorize rows");
        bad_dof_cols_ = safe_alloc<int>(host_bad_dof.cols.size(), rt, "BadDOF gmm factorize cols");
        bad_dof_vals_ = safe_alloc<double>(host_bad_dof.vals.size(), rt, "BadDOF gmm factorize vals");
        bad_dof_rhs_ = safe_alloc<double>(bad_dof_count_, 0.0, rt, "BadDOF gmm factorize rhs");
        bad_dof_solution_ = safe_alloc<double>(bad_dof_count_, 0.0, rt, "BadDOF gmm factorize solution");

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

    void BadDofGMMPreconditioner::apply(
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
