#pragma once

#include <cudss.h>

#include <utility>

namespace polysolve::linear::mas
{
    class CuDSSHandle
    {
    public:
        cudssHandle_t raw = nullptr;

        CuDSSHandle()
        {
            cudssCreate(&raw);
        }

        ~CuDSSHandle()
        {
            if (raw != nullptr)
            {
                cudssDestroy(raw);
            }
        }

        CuDSSHandle(const CuDSSHandle &) = delete;
        CuDSSHandle(CuDSSHandle &&other) { swap(other); }
        CuDSSHandle &operator=(const CuDSSHandle &) = delete;
        CuDSSHandle &operator=(CuDSSHandle &&other)
        {
            swap(other);
            return *this;
        }

    private:
        void swap(CuDSSHandle &other)
        {
            std::swap(raw, other.raw);
        }
    };

    class CuDSSConfig
    {
    public:
        cudssConfig_t raw = nullptr;

        CuDSSConfig()
        {
            cudssConfigCreate(&raw);
        }

        ~CuDSSConfig()
        {
            if (raw != nullptr)
            {
                cudssConfigDestroy(raw);
            }
        }

        CuDSSConfig(const CuDSSConfig &) = delete;
        CuDSSConfig(CuDSSConfig &&other) { swap(other); }
        CuDSSConfig &operator=(const CuDSSConfig &) = delete;
        CuDSSConfig &operator=(CuDSSConfig &&other)
        {
            swap(other);
            return *this;
        }

    private:
        void swap(CuDSSConfig &other)
        {
            std::swap(raw, other.raw);
        }
    };

    class CuDSSData
    {
    public:
        cudssHandle_t owner = nullptr;
        cudssData_t raw = nullptr;

        CuDSSData() = default;

        CuDSSData(cudssHandle_t handle)
        {
            owner = handle;
            cudssDataCreate(handle, &raw);
        }

        ~CuDSSData()
        {
            if (raw != nullptr)
            {
                cudssDataDestroy(owner, raw);
            }
        }

        CuDSSData(const CuDSSData &) = delete;
        CuDSSData(CuDSSData &&other) { swap(other); }
        CuDSSData &operator=(const CuDSSData &) = delete;
        CuDSSData &operator=(CuDSSData &&other)
        {
            swap(other);
            return *this;
        }

    private:
        void swap(CuDSSData &other)
        {
            std::swap(owner, other.owner);
            std::swap(raw, other.raw);
        }
    };

    class CuDSSMatrix
    {
    public:
        cudssMatrix_t raw = nullptr;

        CuDSSMatrix() = default;

        CuDSSMatrix(
            int nrows,
            int ncols,
            int nnz,
            int *row_offsets,
            int *col_indices,
            double *values)
        {
            cudssMatrixCreateCsr(
                &raw,
                nrows,
                ncols,
                nnz,
                row_offsets,
                nullptr,
                col_indices,
                values,
                CUDA_R_32I,
                CUDA_R_64F,
                CUDSS_MTYPE_SYMMETRIC,
                CUDSS_MVIEW_FULL,
                CUDSS_BASE_ZERO);
        }

        CuDSSMatrix(
            int nrows,
            int ncols,
            int ld,
            double *values)
        {
            cudssMatrixCreateDn(
                &raw,
                nrows,
                ncols,
                ld,
                values,
                CUDA_R_64F,
                CUDSS_LAYOUT_COL_MAJOR);
        }

        ~CuDSSMatrix()
        {
            if (raw != nullptr)
            {
                cudssMatrixDestroy(raw);
            }
        }

        CuDSSMatrix(const CuDSSMatrix &) = delete;
        CuDSSMatrix(CuDSSMatrix &&other) { swap(other); }
        CuDSSMatrix &operator=(const CuDSSMatrix &) = delete;
        CuDSSMatrix &operator=(CuDSSMatrix &&other)
        {
            swap(other);
            return *this;
        }

    private:
        void swap(CuDSSMatrix &other)
        {
            std::swap(raw, other.raw);
        }
    };
} // namespace polysolve::linear::mas
