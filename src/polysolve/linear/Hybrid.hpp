#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <vector>
#include <deque>
#include <unordered_map>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#ifdef POLYSOLVE_WITH_ICHOL
#include <boost/property_tree/ptree.hpp>
namespace mschol {
    class ichol_precond;
}
#endif

namespace polysolve::linear
{
    class AbstractSolver {

    public:
        virtual void compute(const Eigen::SparseMatrix<double>& A) = 0;
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& b) = 0;
        virtual ~AbstractSolver() = default;
        virtual void print_nnz() {}
    };

    template <typename EigenSolverT>
    class EigenWrapper : public AbstractSolver {
        EigenSolverT solver;
    public:
        void compute(const Eigen::SparseMatrix<double>& A) override {
            solver.compute(A);
        }

        Eigen::VectorXd solve(const Eigen::VectorXd& b) override {
            return solver.solve(b);
        }
        template <typename T>
        auto print_nnz_impl(T& s, int) -> decltype(s.pardisoParameterArray(), void()) {
            long long fill_in = s.pardisoParameterArray()[17];
            std::cout << "Total LU nnz: " << fill_in << std::endl;
        }

        template <typename T>
        auto print_nnz_impl(T& s, long) -> decltype(s.matrixL(), s.vectorD(), void()) {
            long long fill_in = (2 * s.matrixL().nestedExpression().nonZeros()) + s.vectorD().size();
            std::cout << "Total LU nnz: " << fill_in << std::endl;
        }

        template <typename T>
        void print_nnz_impl(T&, ...) {}

        void print_nnz() override {
            print_nnz_impl(solver, 0);
        }

    };

    class Hybrid : public Solver
    {

    public:
        Hybrid();
        ~Hybrid();

        typedef Eigen::Map<StiffnessMatrix> SharedSparseMatrix;
        typedef Eigen::Map<Eigen::VectorXd> SharedVector;

    private:
        POLYSOLVE_DELETE_MOVE_COPY(Hybrid)

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
                    return "Hybrid";
                }
                return "AMGF";
            }
            if (use_incomplete_cholesky_precond)
            {
                return "Ichol";
            }
            return "AMG"; 
        }

        virtual void set_problematic_dofs(const std::set<int> &bad_dofs) override {all_bad_dofs = bad_dofs;}

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

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 10000;
        double rel_conv_tol_ = 1e-10;
        double abs_conv_tol_ = 0.0;
        double conditioning_threshold = 100.0;
        bool do_mixed_precond = false;
        bool use_incomplete_cholesky_precond = false;

        // solve information
        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;

        // MPI rank distribution 
        int myid = 0;
        int num_procs = 1;
        std::vector<int> starts;
        std::vector<int> ends;

        // problem-specific data
        std::unordered_map<int, Eigen::SparseVector<double>> global_to_row;

        // Hypre variables
        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJVector ij_x;
        HYPRE_IJVector ij_b;

        // hybrid preconditioner data
        std::deque<std::unique_ptr<AbstractSolver>> D_solvers;
        std::set<int> all_bad_dofs;
        std::vector<std::set<int>> bad_indices_sets;
        std::vector<std::vector<int>> bad_indices_arrays;
        std::vector<std::vector<int>> bad_subdomain_assignments;
        std::vector<std::unordered_map<int, int>> index_mappings;

        Eigen::VectorXd shared_rhs, shared_result;
        Eigen::VectorXd z1, z2, z3;
        Eigen::VectorXd r, p, buffer;

#ifdef POLYSOLVE_WITH_ICHOL
        // incomplete cholesky variables
        std::shared_ptr<mschol::ichol_precond> inc_chol_precond; 
        boost::property_tree::ptree pt;
        Eigen::VectorXi ichol_dof_remapping;
        double rho = 3.25;
        int remap_dof(const int index);
        void setup_ichol_precond(SharedSparseMatrix &sparse_A);
#endif

        // factorization helpers
        void partition_ranks(const int rows);
        void copy_matrix_to_hypre(SharedSparseMatrix &sparse_A);
        void copy_matrix_to_hypre(Eigen::SparseMatrix<double> &sparse_A);

        // solve helpers
        void init_hypre_vectors();
        void recover_solution(Eigen::Ref<VectorXd> result);

        // hybrid preconditioner helpers
        void assemble_D(int bad_i, int i, Eigen::SparseMatrix<double>& D, SharedSparseMatrix &sparse_A);
        void build_index_mappings();
        void decompose_subdomains_to_disjoint_subsets(SharedSparseMatrix &sparse_A);
        void filter_subdomains(SharedSparseMatrix &sparse_A);
        void expand_subdomains_to_strongly_connected(SharedSparseMatrix &sparse_A);
        void share_bad_subdomains();
        void load_balance_subdomains();
        void select_bad_dofs(SharedSparseMatrix &sparse_A);
        void factorize_submatrix(SharedSparseMatrix &sparse_A);

        // matrix multiplication
        void matmul(Eigen::VectorXd &x, Eigen::VectorXd &result);
        double dot(Eigen::VectorXd &x, Eigen::VectorXd &y);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z, SharedVector &vec, MPI_Win &vec_win);
        void amg_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd& b, Eigen::VectorXd &x);
        void dss_precond_iter(Eigen::VectorXd &z, Eigen::VectorXd &r, Eigen::VectorXd &next_z, SharedVector &vec, MPI_Win &vec_win);

        // MPI helpers
        void create_shared_vec(MPI_Win &win, void* &base_ptr, int size);
        int my_size() {return ends[myid] - starts[myid] + 1;};

        // Krylov solve methods
        void pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond, SharedVector &vec, MPI_Win &vec_win);
    };

    struct disjointSet {
        std::vector<int> parent, rank;

        disjointSet(int n) 
        {
            rank.assign(n, 0);
            for (int i = 0; i < n; i++)
                parent.push_back(i);
        }

        int find_set(int v) 
        {
            if (parent[v] != v)
                parent[v] = find_set(parent[v]);

            return parent[v];
        }

        void union_set(int x,int y) 
        {
            if (x == y)
            {
                return;
            }

            x = find_set(x);
            y = find_set(y);

            if (rank[x] > rank[y])
                parent[y] = x;
            else
            {
                parent[x] = y;
                if(rank[x] == rank[y])
                    rank[y]++;
            }
        }
    };


} // namespace polysolve::linear
