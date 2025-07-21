# Define your test executable
add_executable(cppgrad_test test_main.cpp)
target_link_libraries(cppgrad_test PRIVATE cppgrad Catch2::Catch2WithMain)

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(cppgrad_test)