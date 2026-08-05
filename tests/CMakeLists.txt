################################################################################
# Tests
################################################################################

set(test_sources
	main.cpp
	test_linear_solver.cpp
	test_nonlinear_solver.cpp
	test_json.cpp
)
add_executable(unit_tests ${test_sources})
add_executable(linear_solve linear_solve.cpp)

################################################################################
# Required Libraries
################################################################################

include(catch2)
target_link_libraries(unit_tests PUBLIC Catch2::Catch2)

target_link_libraries(unit_tests PUBLIC polysolve::polysolve)
target_link_libraries(linear_solve PUBLIC polysolve::polysolve)

include(polysolve_warnings)
target_link_libraries(unit_tests PRIVATE polysolve::warnings)
target_link_libraries(linear_solve PRIVATE polysolve::warnings)

if(DEFINED argparse_SOURCE_DIR)
    target_include_directories(linear_solve SYSTEM PRIVATE "${argparse_SOURCE_DIR}/include")
endif()

if(POLYSOLVE_WITH_SANITIZERS)
	add_sanitizers(unit_tests)
	add_sanitizers(linear_solve)
endif()

if(POLYSOLVE_WITH_AMGCL)
    target_compile_definitions(unit_tests PRIVATE -DPOLYSOLVE_WITH_AMGCL)
endif()

include(polyfem-data)
target_link_libraries(unit_tests PRIVATE polyfem::data)

################################################################################
# Register tests
################################################################################

foreach(source IN ITEMS ${test_sources})
    source_group("tests" FILES "${source}")
endforeach()
source_group("tests" FILES "linear_solve.cpp")

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(unit_tests)
