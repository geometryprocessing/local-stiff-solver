if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(blas)
include(boost)

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG 98481a9b270fb8e6291b82ee498e28ead31923fd
)

# mkl provides mkl_cblas.h instead of cblas.h.
# create compatibility shim.
if(POLYSOLVE_WITH_MKL)
    set(mschol_mkl_compat_include_dir "${CMAKE_CURRENT_BINARY_DIR}/mschol_mkl_compat")
    file(MAKE_DIRECTORY "${mschol_mkl_compat_include_dir}")
    file(WRITE "${mschol_mkl_compat_include_dir}/cblas.h" "#pragma once\n#include <mkl_cblas.h>\n")
    target_include_directories(mschol SYSTEM PRIVATE "${mschol_mkl_compat_include_dir}")
endif()

target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)
