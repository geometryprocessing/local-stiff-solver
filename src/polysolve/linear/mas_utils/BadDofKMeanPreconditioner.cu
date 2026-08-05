#include <polysolve/linear/mas_utils/BadDofKMeanPreconditioner.hpp>

#include <cuda/algorithm>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cmath>
#include <utility>
#include <vector>
#include <array>

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

        std::vector<int> select_bad_dofs_from_diagonal_kmeans(
            const std::vector<double> &off_diag_max_abs,
            double search_fraction,
            double jump_threshold,
            int max_iterations,
            int expand_neighbors)
        {
            int n = off_diag_max_abs.size();
            if (n < 2)
            {
                return {};
            }

            // Sort index value pair in descending order.
            std::vector<std::pair<int, double>> entries;
            for (int i = 0; i < n; ++i)
            {
                entries.emplace_back(i, off_diag_max_abs[i]);
            }
            std::sort(entries.begin(), entries.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                return a.second > b.second;
            });

            assert(search_fraction > 0.0 && search_fraction <= 1.0);

            // Cluster top fraction dof only.
            int subset_num = std::clamp(static_cast<int>(n * search_fraction), 2, n);
            ctd::span<std::pair<int, double>> subset{entries.data(), entries.data() + subset_num};

            // K-mean clustering via Lloyd's algorithm.
            // Since k == 2, choose start and end as initial cluster center.

            // cluster center
            std::array<double, 2> c = {subset.front().second,
                                       subset.back().second};
            assert(max_iterations > 0);
            for (int i = 0; i < max_iterations; ++i)
            {
                std::array<double, 2> c_sum = {0.0, 0.0};
                std::array<int, 2> c_count = {0, 0};

                for (auto [idx, val] : subset)
                {
                    int cid = static_cast<int>(std::abs(val - c[0]) < std::abs(val - c[1]));
                    c_count[cid]++;
                    c_sum[cid] += val;
                }

                std::array<double, 2> c_next;
                c_next[0] = (c_count[0] > 0) ? (c_sum[0] / c_count[0]) : c[0];
                c_next[1] = (c_count[1] > 1) ? (c_sum[1] / c_count[1]) : c[1];
                if (c_next == c)
                {
                    break;
                }

                c = c_next;
            }

            // Test ratio c_large / c_small.
            if (c[0] < c[1])
            {
                std::swap(c[0], c[1]);
            }
            double ratio = c[0] / c[1];
            if (std::isnan(ratio) || ratio < jump_threshold)
            {
                return {};
            }

            // gather results.
            double sep = 0.5 * (c[0] + c[1]);
            if (expand_neighbors <= 0)
            {
                std::vector<int> selected_positions;
                for (int i = 0; i < subset_num; ++i)
                {
                    if (subset[i].second > sep)
                    {
                        selected_positions.push_back(subset[i].first);
                    }
                }
                return selected_positions;
            }

            // Expand selection by neighboring indices.
            std::vector<unsigned char> selected_mask(n, 0); // avoid vector<bool> specialization.
            std::vector<int> selected_positions;
            for (int i = 0; i < subset_num; ++i)
            {
                if (subset[i].second <= sep)
                {
                    continue;
                }

                int pos = subset[i].first;
                int l = std::max(0, pos - expand_neighbors);
                int r = std::min(n - 1, pos + expand_neighbors);
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
            double kmeans_search_fraction,
            double kmeans_jump_threshold,
            int kmeans_max_iterations,
            int kmeans_expand_neighbors,
            CudaRuntime rt)
        {
            HostBadDofMatrix out;
            auto h_rows = copy_device_span_to_host(view.rows, rt);
            auto h_cols = copy_device_span_to_host(view.cols, rt);
            auto h_vals = copy_device_span_to_host(view.vals, rt);

            // Gather permuted max abs(off-diagonal) excluding BSR padding.
            int block_size = view.block_dim * view.block_dim;
            int permuted_scalar_dim = view.dim * view.block_dim;
            std::vector<double> off_diag_max_abs;
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
                            if (scalar_col == scalar_row)
                            {
                                continue;
                            }

                            val = std::max(val, std::abs(h_vals[row_offset + local_col]));
                        }
                    }

                    diag_idx.push_back(scalar_row);
                    off_diag_max_abs.push_back(val);
                }
            }

            // Select bad dof and build gather/scatter map.
            std::vector<int> selected_positions = select_bad_dofs_from_diagonal_kmeans(
                off_diag_max_abs,
                kmeans_search_fraction,
                kmeans_jump_threshold,
                kmeans_max_iterations,
                kmeans_expand_neighbors);
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

    void BadDofKMeanPreconditioner::reset()
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

    void BadDofKMeanPreconditioner::factorize(
        const BSRMatrix &A,
        CudaRuntime rt)
    {
        auto phase_begin = clock::now();
        reset();
        cudssSetStream(cudss_handle_.raw, rt.stream.get());

        BSRView view = A.view();
        vector_size_ = view.dim * view.block_dim;

        HostBadDofMatrix host_bad_dof = build_bad_dof_matrix(
            view,
            kmeans_search_fraction_,
            kmeans_jump_threshold_,
            kmeans_max_iterations_,
            kmeans_expand_neighbors_,
            rt);
        bad_dof_count_ = host_bad_dof.indices.size();
        SPDLOG_INFO(
            "[MAS] [factorize_bad_dof_kmeans] [{:.6f}] [count={}]",
            elapsed_seconds(phase_begin),
            bad_dof_count_);
        if (bad_dof_count_ == 0)
        {
            return;
        }

        bad_dof_indices_ = safe_alloc<int>(host_bad_dof.indices.size(), rt, "BadDOF kmeans factorize indices");
        bad_dof_rows_ = safe_alloc<int>(host_bad_dof.rows.size(), rt, "BadDOF kmeans factorize rows");
        bad_dof_cols_ = safe_alloc<int>(host_bad_dof.cols.size(), rt, "BadDOF kmeans factorize cols");
        bad_dof_vals_ = safe_alloc<double>(host_bad_dof.vals.size(), rt, "BadDOF kmeans factorize vals");
        bad_dof_rhs_ = safe_alloc<double>(bad_dof_count_, 0.0, rt, "BadDOF kmeans factorize rhs");
        bad_dof_solution_ = safe_alloc<double>(bad_dof_count_, 0.0, rt, "BadDOF kmeans factorize solution");

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

    void BadDofKMeanPreconditioner::apply(
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
