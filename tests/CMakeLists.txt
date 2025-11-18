cmake_minimum_required(VERSION 3.20)

# Test executable for C++ language model
add_executable(test_language_model
    cpp/test_language_model.cpp
)

# Link test executable with the library (this brings in Eigen3 if available)
target_link_libraries(test_language_model language_model_lib)

# Add test to CTest
add_test(NAME LanguageModelTest COMMAND test_language_model)
