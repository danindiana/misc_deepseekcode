/**
 * Language Model Example - Main Program
 * Demonstrates object-oriented design for language models
 */

#include "../../src/cpp/language_model.h"

int main() {
    InterSystemCommunicationLanguage system;
    system.addLanguageModel(std::make_unique<StatisticalLanguageModel>());
    system.addLanguageModel(std::make_unique<NeuralNetworkLanguageModel>());

    std::string inputText = "Hello, how are you?";
    auto rawResponses = system.communicate(inputText);
    auto optimizedResponse = system.optimizeCommunication(rawResponses);

    std::cout << "Optimized Response: " << optimizedResponse << std::endl;

    return 0;
}
