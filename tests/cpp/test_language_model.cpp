/**
 * Unit tests for C++ Language Model implementation
 * Compile with: g++ -std=c++20 -o test_language_model test_language_model.cpp
 */

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <memory>

// Include the implementation
// In a real project, this would be a proper header include
#include "../../examples/cpp/language_model.cpp"

void test_statistical_model_creation() {
    std::cout << "Testing StatisticalLanguageModel creation..." << std::endl;
    auto model = std::make_unique<StatisticalLanguageModel>();
    assert(model != nullptr);
    std::cout << "✓ StatisticalLanguageModel created successfully" << std::endl;
}

void test_neural_model_creation() {
    std::cout << "Testing NeuralNetworkLanguageModel creation..." << std::endl;
    auto model = std::make_unique<NeuralNetworkLanguageModel>();
    assert(model != nullptr);
    std::cout << "✓ NeuralNetworkLanguageModel created successfully" << std::endl;
}

void test_statistical_model_response() {
    std::cout << "Testing StatisticalLanguageModel response generation..." << std::endl;
    auto model = std::make_unique<StatisticalLanguageModel>();
    std::string input = "Hello, world!";
    std::string response = model->generateResponse(input);

    assert(!response.empty());
    assert(response.find("Statistical response") != std::string::npos);
    std::cout << "✓ Statistical model generates valid response" << std::endl;
}

void test_neural_model_response() {
    std::cout << "Testing NeuralNetworkLanguageModel response generation..." << std::endl;
    auto model = std::make_unique<NeuralNetworkLanguageModel>();
    std::string input = "Hello, world!";
    std::string response = model->generateResponse(input);

    assert(!response.empty());
    assert(response.find("Neural network response") != std::string::npos);
    std::cout << "✓ Neural network model generates valid response" << std::endl;
}

void test_communication_system_initialization() {
    std::cout << "Testing InterSystemCommunicationLanguage initialization..." << std::endl;
    InterSystemCommunicationLanguage system;
    std::cout << "✓ Communication system initialized successfully" << std::endl;
}

void test_add_models() {
    std::cout << "Testing adding models to communication system..." << std::endl;
    InterSystemCommunicationLanguage system;

    system.addLanguageModel(std::make_unique<StatisticalLanguageModel>());
    system.addLanguageModel(std::make_unique<NeuralNetworkLanguageModel>());

    std::cout << "✓ Models added successfully" << std::endl;
}

void test_communicate() {
    std::cout << "Testing communication between models..." << std::endl;
    InterSystemCommunicationLanguage system;

    system.addLanguageModel(std::make_unique<StatisticalLanguageModel>());
    system.addLanguageModel(std::make_unique<NeuralNetworkLanguageModel>());

    std::string input = "Test input";
    auto responses = system.communicate(input);

    assert(responses.size() == 2);
    assert(!responses[0].empty());
    assert(!responses[1].empty());
    std::cout << "✓ Communication works correctly" << std::endl;
}

void test_optimize_communication() {
    std::cout << "Testing communication optimization..." << std::endl;
    InterSystemCommunicationLanguage system;

    system.addLanguageModel(std::make_unique<StatisticalLanguageModel>());
    system.addLanguageModel(std::make_unique<NeuralNetworkLanguageModel>());

    std::string input = "Test input";
    auto responses = system.communicate(input);
    auto optimized = system.optimizeCommunication(responses);

    assert(!optimized.empty());
    std::cout << "✓ Communication optimization works" << std::endl;
}

void test_empty_responses() {
    std::cout << "Testing optimization with empty responses..." << std::endl;
    InterSystemCommunicationLanguage system;

    std::vector<std::string> empty_responses;
    auto optimized = system.optimizeCommunication(empty_responses);

    assert(optimized == "No responses");
    std::cout << "✓ Empty responses handled correctly" << std::endl;
}

int main() {
    std::cout << "=== Running C++ Language Model Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_statistical_model_creation();
        test_neural_model_creation();
        test_statistical_model_response();
        test_neural_model_response();
        test_communication_system_initialization();
        test_add_models();
        test_communicate();
        test_optimize_communication();
        test_empty_responses();

        std::cout << std::endl;
        std::cout << "=== All Tests Passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}
