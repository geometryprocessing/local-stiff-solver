#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include "Hybrid.hpp"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#include <cudss.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

extern "C" {
    HYPRE_Int hypre_ParVectorAxpy(HYPRE_Complex alpha, HYPRE_ParVector x, HYPRE_ParVector y);
}

namespace polysolve::linear
{

    class GPUHybrid : public Solver
    {

    public:
        GPUHybrid();
        ~GPUHybrid();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(GPUHybrid)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve solve information
        virtual void get_info(json &params) const override;

        void check_settings() const;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { 
            if (do_mixed_precond)
            {
                if (select_bad_dofs_from_l1_row_norm)
                {
                    return "GPUHybrid";
                }
                return "GPUAMGF";
            }
            return "GPUAMG"; 
        }

        virtual void set_problematic_dofs(const std::set<int> &bad_dofs) override {h_all_bad_dofs = bad_dofs;}

    protected:
        // AMG settings
        double theta = 0.5;

        // Hybrid preconditioner settings
        bool select_bad_dofs_from_l1_row_norm = true;
        bool decompose_subdomains = false;
        int min_subdomain_size = 1;
        int max_subdomain_size = 1e9;
        double gmm_jump_threshold = 100.0;
        double gmm_tol = 1e-3;
        int max_gmm_iterations = 20;
        bool expand_subdomains = true;
        int max_dense_size = 0;
        bool use_float_on_subdomains = false;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 10000;
        double rel_conv_tol_ = 1e-10;
        double abs_conv_tol_ = 0.0;
        double conditioning_threshold = 100.0;
        bool do_mixed_precond = false;

        // solve information
        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;

        // Hypre variables
        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJVector ij_x;
        HYPRE_IJVector ij_b;

        // hybrid preconditioner data
        std::set<int> h_all_bad_dofs;
        std::vector<int> h_subdomain_sizes;
        thrust::device_vector<int> d_subdomain_sizes;
        thrust::device_vector<int> d_all_bad_dofs;
        std::vector<std::vector<int>> bad_indices_arrays;

        thrust::device_vector<int> d_inner_indices;
        thrust::device_vector<int> d_outer_indices;
        thrust::device_vector<double> d_values;

        cudssHandle_t cudss_handle = nullptr;
        cudssConfig_t cudss_config = nullptr;
        cudssData_t cudss_solver_data = nullptr;
        
        cudssMatrix_t batch_A = nullptr;
        cudssMatrix_t batch_x = nullptr;
        cudssMatrix_t batch_b = nullptr;

        cublasHandle_t cublas_handle = nullptr;

        int dense_batch_count = 0;
        int max_dense_dim = 0;
        int sparse_batch_count = 0;

        thrust::device_vector<int> d_sparse_dof_map;
        thrust::device_vector<int> d_dense_dof_map;

        // dense solve data
        thrust::device_vector<double> d_dense_invs;
        thrust::device_vector<double*> d_dense_ptrs;
        thrust::device_vector<int> d_pivots;
        thrust::device_vector<int> d_info;
        thrust::device_vector<double> d_dense_x;
        thrust::device_vector<double*> d_dense_x_ptrs;
        thrust::device_vector<double> d_dense_b;
        thrust::device_vector<double*> d_dense_b_ptrs;
        thrust::device_vector<int> d_dense_dof_offsets;
        thrust::device_vector<int> d_dense_true_sizes;

        std::vector<int> h_dense_dof_starts;

        // sparse solve data
        std::vector<int> h_sparse_nrows, h_sparse_ncols, h_sparse_nnz, h_sparse_vec_ncols, h_sparse_ld;

        thrust::device_vector<int> d_sparse_inner_indices, d_sparse_outer_indices;
        thrust::device_vector<double> d_sparse_values;
        thrust::device_vector<float> d_sparse_single_values;
        thrust::device_vector<double> d_sparse_x, d_sparse_b;
        thrust::device_vector<float> d_sparse_single_x, d_sparse_single_b;
        thrust::device_vector<void*> d_sparse_inner_void, d_sparse_outer_void, d_sparse_values_void;
        thrust::device_vector<void*> d_sparse_x_void, d_sparse_b_void;

    public:
        void free_device_memory();

        // factorization helpers
        void copy_matrix_to_hypre();

        // solve helpers
        void init_hypre_vectors(const int size);

        // hybrid preconditioner helpers
        void decompose_subdomains_to_disjoint_subsets(const Eigen::SparseMatrix<double>& sparse_A);
        void filter_subdomains(const Eigen::SparseMatrix<double>& sparse_A);
        void expand_subdomains_to_strongly_connected(const Eigen::SparseMatrix<double>& sparse_A);
        void select_bad_dofs();
        void factorize_submatrix();

        // linear algebra
        void set_hypre_vec(HYPRE_IJVector &ij_x, HYPRE_ParVector &par_x, const thrust::device_vector<double>& x);
        void matmul(const thrust::device_vector<double>& x, thrust::device_vector<double>& result);
        double dot(const thrust::device_vector<double>& x, const thrust::device_vector<double>& y);
        void vector_copy(const thrust::device_vector<double>& x, thrust::device_vector<double>& y);
        void vector_add(double alpha, const thrust::device_vector<double>& x, thrust::device_vector<double>& y);
        void vector_scale(double alpha, thrust::device_vector<double>& x);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &r, thrust::device_vector<double> &z, thrust::device_vector<double> &buffer, thrust::device_vector<double> &z2);
        void amg_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double>& b, thrust::device_vector<double> &x);
        void dss_precond_iter(thrust::device_vector<double> &z, thrust::device_vector<double> &r, thrust::device_vector<double> &next_z);

        // Krylov solve methods
        void pcg_solve(thrust::device_vector<double> &rhs, thrust::device_vector<double> &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond);
    };

} // namespace polysolve::linear