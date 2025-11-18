/**
 * Language Model Header File
 * Declarations for language model classes
 */

#ifndef LANGUAGE_MODEL_H
#define LANGUAGE_MODEL_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>

// Forward declaration of LanguageModel
class LanguageModel {
public:
    virtual ~LanguageModel() = default;
    virtual std::string generateResponse(const std::string& inputText) = 0;
};

// Statistical language model implementation
class StatisticalLanguageModel : public LanguageModel {
public:
    StatisticalLanguageModel();
    std::string generateResponse(const std::string& inputText) override;
};

// Neural network language model implementation
class NeuralNetworkLanguageModel : public LanguageModel {
public:
    NeuralNetworkLanguageModel();
    std::string generateResponse(const std::string& inputText) override;
};

// InterSystemCommunicationLanguage class
class InterSystemCommunicationLanguage {
private:
    std::vector<std::unique_ptr<LanguageModel>> languageModels;

public:
    void addLanguageModel(std::unique_ptr<LanguageModel> model);
    std::vector<std::string> communicate(const std::string& inputText);
    std::string optimizeCommunication(const std::vector<std::string>& responses);
};

#endif // LANGUAGE_MODEL_H
