# Find OpenMP
find_package(OpenMP REQUIRED)

# Create executable
add_executable(openmp_test single_iterators.cpp)

# Link OpenMP
target_link_libraries(openmp_test OpenMP::OpenMP_CXX)