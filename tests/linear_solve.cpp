#include <polysolve/Types.hpp>
#include <polysolve/linear/Solver.hpp>

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <unsupported/Eigen/SparseExtra>
#include <_hypre_utilities.h>

#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <sys/resource.h>

#ifdef HYPRE_ENABLE_MPI
#include <mpi.h>
#endif

using namespace polysolve;
using namespace polysolve::linear;

namespace
{
    constexpr std::uint32_t DEFAULT_RAND_SEED = 0;

    class ScopedOutputSilencer
    {
    public:
        ScopedOutputSilencer()
        {
            flush_output();

            null_fd_ = open("/dev/null", O_WRONLY);
            if (null_fd_ == -1)
            {
                throw std::runtime_error("failed to open /dev/null");
            }

            stdout_fd_ = dup(STDOUT_FILENO);
            if (stdout_fd_ == -1)
            {
                close(null_fd_);
                throw std::runtime_error("failed to duplicate stdout");
            }

            stderr_fd_ = dup(STDERR_FILENO);
            if (stderr_fd_ == -1)
            {
                close(stdout_fd_);
                close(null_fd_);
                throw std::runtime_error("failed to duplicate stderr");
            }

            if (dup2(null_fd_, STDOUT_FILENO) == -1 || dup2(null_fd_, STDERR_FILENO) == -1)
            {
                restore();
                throw std::runtime_error("failed to redirect output to /dev/null");
            }
        }

        ~ScopedOutputSilencer()
        {
            restore();
        }

        ScopedOutputSilencer(const ScopedOutputSilencer &) = delete;
        ScopedOutputSilencer &operator=(const ScopedOutputSilencer &) = delete;

    private:
        void restore() noexcept
        {
            flush_output();

            if (stdout_fd_ != -1)
            {
                (void)dup2(stdout_fd_, STDOUT_FILENO);
                close(stdout_fd_);
                stdout_fd_ = -1;
            }
            if (stderr_fd_ != -1)
            {
                (void)dup2(stderr_fd_, STDERR_FILENO);
                close(stderr_fd_);
                stderr_fd_ = -1;
            }
            if (null_fd_ != -1)
            {
                close(null_fd_);
                null_fd_ = -1;
            }
        }

        static void flush_output() noexcept
        {
            std::fflush(stdout);
            std::fflush(stderr);
        }

        int null_fd_ = -1;
        int stdout_fd_ = -1;
        int stderr_fd_ = -1;
    };

    Eigen::VectorXd make_random_rhs(const Eigen::Index size, const std::optional<std::uint32_t> seed)
    {
        std::mt19937 generator;
        if (seed.has_value())
        {
            generator.seed(*seed);
        }
        else
        {
            std::random_device random_device;
            generator.seed(random_device());
        }

        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        Eigen::VectorXd rhs(size);
        for (Eigen::Index i = 0; i < size; ++i)
        {
            rhs[i] = distribution(generator);
        }
        return rhs;
    }

    template <typename Derived>
    bool loadMarketDense(Eigen::PlainObjectBase<Derived>& mat, const std::string& filename) {
        std::ifstream in(filename);
        if (!in) return false;

        std::string line;
        int rows = 0, cols = 0;

        while (std::getline(in, line)) {
            size_t start = line.find_first_not_of(" \t\r\n");
            
            if (start == std::string::npos || line[start] == '%') {
                continue;
            }
            
            std::stringstream ss(line.substr(start));
            ss >> rows >> cols;
            
            int nonzeros;
            if (ss >> nonzeros) {
                std::cerr << "Error: File appears to be coordinate format, expected array." << "\n";
                return false;
            }
            break;
        }

        mat.resize(rows, cols);

        for (int j = 0; j < mat.cols(); ++j) {
            for (int i = 0; i < mat.rows(); ++i) {
                if (!(in >> mat(i, j))) {
                    std::cerr << "Error: Unexpected end of file or invalid data format." << "\n";
                    return false; 
                }
            }
        }
        
        return true;
    }

    size_t getPeakRSS()
    {
        struct rusage rusage;
        getrusage(RUSAGE_SELF, &rusage);
        return (size_t)(rusage.ru_maxrss * 1024L);
    }
} // namespace

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("linear_solve");
    program.add_description("Run a PolySolve linear solve from Matrix Market inputs.");
    program.add_argument("-A")
        .required()
        .metavar("matrix.mtx")
        .help("Input Matrix Market sparse matrix.");
    auto &rhs_group = program.add_mutually_exclusive_group();
    rhs_group.add_argument("-b")
        .metavar("rhs.mtx")
        .help("Optional Matrix Market RHS vector. Defaults to a deterministic random vector with seed 0 unless --rand is used.");
    rhs_group.add_argument("--rand")
        .metavar("seed")
        .scan<'i', long long>()
        .nargs(argparse::nargs_pattern::optional)
        .help("Generate a random RHS instead of loading -b. Optionally provide a seed.");
    program.add_argument("-i")
        .metavar("problem_info.mtx")
        .help("Optional Matrix Market vector of problem specific information.");
    program.add_argument("-t", "--threshold")
        .help("Threshold value to apply.")
        .scan<'g', double>()
        .default_value(0.0);
    program.add_argument("--less_than")
        .default_value(false)
        .implicit_value(true)
        .help("Condition to select problematic DoFs relative to threshold. (Defaults to problematic_info(i) > thresh.)");
    program.add_argument("-p")
        .metavar("positions.mtx")
        .help("Optional Matrix Market file specifiying positions.");
    program.add_argument("-e")
        .metavar("elements.mtx")
        .help("Optional Matrix Market file specifiying connectivity.");
    program.add_argument("-j")
        .metavar("spec.json")
        .help("Optional solver JSON config. Defaults to {\"solver\":\"Eigen::SimplicialLDLT\"}.");
    program.add_argument("-w")
        .default_value(1)
        .scan<'i', int>()
        .metavar("warmup")
        .help("Number of warmup iterations. Default: 1.");
    program.add_argument("-r")
        .default_value(1)
        .scan<'i', int>()
        .metavar("repeat")
        .help("Number of iterations after warmup. Default: 1.");
    program.add_argument("--force_symmetry")
        .default_value(false)
        .implicit_value(true)
        .help("Force matrix to be symmetric. Default: false.");

    program.parse_args(argc, argv);

    const std::string matrix_path = program.get<std::string>("-A");
    const int warmup = program.get<int>("-w");
    const int repeat = program.get<int>("-r");
    const bool use_random_rhs = program.is_used("--rand");
    const auto rand_seed_argument = program.present<long long>("--rand");
    const bool force_symmetry = program.get<bool>("--force_symmetry");

    if (warmup < 0)
    {
        throw std::runtime_error("warmup count must be non-negative");
    }
    if (repeat < 1)
    {
        throw std::runtime_error("repeat count must be at least 1");
    }

    std::optional<std::uint32_t> random_seed;
    if (rand_seed_argument.has_value())
    {
        if (*rand_seed_argument < 0)
        {
            throw std::runtime_error("random seed must be non-negative");
        }
        if (static_cast<unsigned long long>(*rand_seed_argument) > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("random seed exceeds uint32_t range");
        }
        random_seed = static_cast<std::uint32_t>(*rand_seed_argument);
    }

    json solver_config = json::object({{"solver", "Eigen::SimplicialLDLT"}});
    if (program.present("-j"))
    {
        const std::string json_path = program.get<std::string>("-j");
        std::ifstream in(json_path);
        if (!in.is_open())
        {
            throw std::runtime_error("failed to open json config: " + json_path);
        }
        in >> solver_config;
    }

#ifdef HYPRE_ENABLE_MPI
    int done_already;
    int myid = 0, num_procs = 1;

    MPI_Initialized(&done_already);
    if (!done_already)
    {
        MPI_Init(&argc, &argv);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    HYPRE_Initialize();

    auto logger = spdlog::stderr_color_mt("linear_solve");
    logger->set_level(spdlog::level::off);

    // Validate solver configuration before loading potentially large inputs.
    {
        auto solver = Solver::create(solver_config, *logger);
        (void)solver;
    }

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd x, b;

#ifdef HYPRE_ENABLE_MPI
    if (myid != 0)
    {
        auto solver = Solver::create(solver_config, *logger);
        while (true)
        {
            solver->analyze_pattern(A, 0);
            solver->factorize(A);
            solver->solve(b, x);
        }
    }
#endif

    if (!loadMarket(A, matrix_path))
    {
        throw std::runtime_error("failed to load matrix market matrix: " + matrix_path);
    }
    if (A.rows() != A.cols())
    {
        throw std::runtime_error("matrix must be square");
    }
    if (force_symmetry)
    {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(2 * (A.nonZeros() - A.rows()) + A.rows());

        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
            {   
                triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
                if (it.col() != it.row())
                {
                    triplets.push_back(Eigen::Triplet<double>(it.col(), it.row(), it.value()));
                }
            }
        }

        A.setFromTriplets(triplets.begin(), triplets.end());
    }
    A.makeCompressed();

    b.resize(A.rows());
    if (program.present("-b"))
    {
        const std::string rhs_path = program.get<std::string>("-b");
        if (!Eigen::loadMarketVector(b, rhs_path))
        {
            throw std::runtime_error("failed to load matrix market rhs: " + rhs_path);
        }
        if (b.size() != A.rows())
        {
            throw std::runtime_error("rhs dimension mismatch");
        }
    }
    else
    {
        if (use_random_rhs)
        {
            b = make_random_rhs(A.rows(), random_seed);
        }
        else
        {
            b = make_random_rhs(A.rows(), DEFAULT_RAND_SEED);
        }
    }

    Eigen::MatrixXd positions;
    Eigen::MatrixXi elements;

    if (program.present("-p"))
    {
        const std::string positions_path = program.get<std::string>("-p");
        if (!loadMarketDense(positions, positions_path))
        {
            throw std::runtime_error("failed to load matrix market positions: " + positions_path);
        }
    }

    if (program.present("-e"))
    {
        const std::string elements_path = program.get<std::string>("-e");
        if (!loadMarketDense(elements, elements_path))
        {
            throw std::runtime_error("failed to load matrix market elements: " + elements_path);
        }
    }

    std::set<int> problem_specific_bad_dofs;
    if (program.present("-i"))
    {
        const bool less_than = program.get<bool>("--less_than");
        const double problematic_threshold = program.get<double>("-t");
        const std::string p_path = program.get<std::string>("-i");
        Eigen::VectorXd problematic_info(A.rows());

        if (!Eigen::loadMarketVector(problematic_info, p_path))
        {
            throw std::runtime_error("failed to load matrix market problem info: " + p_path);
        }
        if (problematic_info.size() != A.rows())
        {
            throw std::runtime_error("problematic info dimension mismatch");
        }

        for (int i = 0; i < problematic_info.size(); ++i)
        {
            if (less_than)
            {
                if (problematic_info(i) < problematic_threshold)
                {
                    problem_specific_bad_dofs.insert(i);
                }
            }
            else
            {
                if (problematic_info(i) > problematic_threshold)
                {
                    problem_specific_bad_dofs.insert(i);
                }
            }
        }
    }

    std::optional<Eigen::MatrixXd> dense_A;

    const int iterations = warmup + repeat;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const std::unique_ptr<ScopedOutputSilencer> silencer =
            iteration < warmup ? std::make_unique<ScopedOutputSilencer>() : nullptr;

        auto solver = Solver::create(solver_config, *logger);
        solver->set_problematic_dofs(problem_specific_bad_dofs);
        solver->set_positions(positions);
        solver->set_elements(elements);
        SPDLOG_INFO("[{}] [matrix_info] [size={}] [nnzs={}]", solver->name(), A.rows(), A.nonZeros());
        x.resize(A.cols());
        x.setZero();

        if (solver->is_dense())
        {
            if (!dense_A.has_value())
            {
                dense_A.emplace(A);
            }
            solver->analyze_pattern_dense(*dense_A, A.rows());
        }
        else
        {
            SPDLOG_INFO("[{}] [start_analyze_pattern]", solver->name());
            solver->analyze_pattern(A, A.rows());
        }

        if (solver->is_dense())
        {
            solver->factorize_dense(*dense_A);
        }
        else
        {
            SPDLOG_INFO("[{}] [start_factorize]", solver->name());
            solver->factorize(A);
        }

        SPDLOG_INFO("[{}] [start_solve]", solver->name());
        solver->solve(b, x);
        double residual = (b - A * x).norm();
        SPDLOG_INFO("[{}] [residual={}] [peak_memory={}]", solver->name(), residual, getPeakRSS());
    }

#ifdef HYPRE_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
#endif
    HYPRE_Finalize();

    return 0;
}
