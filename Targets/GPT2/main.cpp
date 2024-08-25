//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "BPE.hpp"
#include "Model.hpp"
// External includes
#include <aix.hpp>
#include <aixDevices.hpp>
#include <docopt/docopt.h>
// System includes
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>


enum class ModelConfigType : size_t
{
    OPENAI_124M  = 0,
    OPENAI_355M  = 1,
    OPENAI_774M  = 2,
    OPENAI_1558M = 3,
};

struct CmdLineOptions
{
    std::string prompt;
    ModelConfigType modelType{ModelConfigType::OPENAI_124M};
};

CmdLineOptions processCommandLineArguments(int argc, const char* argv[])
{
    static const char USAGE[] =
    R"(
    GPT2 - Copyright (c) 2024-Present, Arkin Terli. All rights reserved.

    Usage:
        GPT2 --prompt=<text> --model=<type>

    Example:
        GPT2 --prompt="What do you know about artificial intelligence?" --model=124M

    Options:
        --prompt=<text>     Your prompt to the GPT2.
        --model=<type>      Model type to use. Options: [124M | 355M | 774M | 1558M]
    )";

    std::map <std::string, docopt::value>  args;
    CmdLineOptions options;

    try
    {
        // Check cmd-line parameters.
        std::vector<std::string>  baseArgs{ argv + 1, argv + argc };

        // Parse cmd-line parameters.
        args = docopt::docopt(USAGE, {argv + 1, argv + argc}, false, "GPT2 0.0.0");

        options.prompt = args["--prompt"].asString();
        auto modelType = args["--model"].asString();

        if (options.prompt.empty()) throw std::invalid_argument("Prompt cannot be empty.");

        if (modelType == "124M")        options.modelType = ModelConfigType::OPENAI_124M;
        else if (modelType == "355M")   options.modelType = ModelConfigType::OPENAI_355M;
        else if (modelType == "774M")   options.modelType = ModelConfigType::OPENAI_774M;
        else if (modelType == "1558M")  options.modelType = ModelConfigType::OPENAI_1558M;
        else
            throw std::invalid_argument("Unknown model parameter type: " + modelType);
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception message: " << e.what() << std::endl;
        exit(-1);
    }

    return options;
}

void validateFileExistence(const std::string& filePath)
{
    if (!std::filesystem::exists(filePath))
    {
        std::cerr << "File does not exist: " + filePath << std::endl;
        exit(-1);
    }
}


int main(int argc, const char* argv[])
{
    // NOTE: All the configuration is prepared here instead of using a separate config file to reduce noise
    //       only for the example purpose.

    // GPT2 model parameters.
    // nVocab : Number of tokens in our vocabulary.
    // nCtx   : Maximum possible token sequence length for the input.
    // nEmbd  : Embedding dimension (determines the "width" of the network).
    // nHeads : Number of attention heads (embedding dimension must be divisible by heads).
    // nLayers: Number of layers (determines the "depth" of the network).
    std::vector<std::unordered_map<std::string, size_t>>  modelParams
    {
        { {"nVocab", 50257}, {"nCtx", 1024}, {"nEmbd",  768}, {"nHeads", 12}, {"nLayers", 12}, },   // 124M
        { {"nVocab", 50257}, {"nCtx", 1024}, {"nEmbd", 1024}, {"nHeads", 16}, {"nLayers", 24}, },   // 355M
        { {"nVocab", 50257}, {"nCtx", 1024}, {"nEmbd", 1280}, {"nHeads", 20}, {"nLayers", 36}, },   // 774M
        { {"nVocab", 50257}, {"nCtx", 1024}, {"nEmbd", 1600}, {"nHeads", 25}, {"nLayers", 48}, },   // 1558M

    };

    std::vector<std::string> modelWeightsFilenames =
    {
        "Resources/GPT2/oaiWeights124M.bin",   // 124M parameters.
        "Resources/GPT2/oaiWeights355M.bin",   // 355M parameters.
        "Resources/GPT2/oaiWeights774M.bin",   // 774M parameters.
        "Resources/GPT2/oaiWeights1558M.bin",  // 1558M parameters.
    };

    // Get command-line options.
    auto cmdLineOptions = processCommandLineArguments(argc, argv);

    // Configurations.
    auto modelType    = static_cast<size_t>(cmdLineOptions.modelType);
    auto hParams      = modelParams[modelType];
    auto modelFile    = modelWeightsFilenames[modelType];
    auto bpeMergeFile = "Resources/GPT2/oaiBPEMergeRules.txt";
    auto bpeVocabFile = "Resources/GPT2/oaiBPEVocabs.txt";
    auto deviceType   = aix::DeviceType::kGPU_METAL;

    std::string prompt = cmdLineOptions.prompt;

    // Check if all the necessary files do exist.
    validateFileExistence(bpeMergeFile);
    validateFileExistence(bpeVocabFile);
    validateFileExistence(modelFile);

    // -----------------------------------------------------------
    // Create a model, and process the prompt.
    // -----------------------------------------------------------

    // Create a BPE, Byte-Pair-Encoding tokenizer.
    BPE bpe(bpeMergeFile, bpeVocabFile);

    // Create a device that uses Apple Metal for GPU computations.
    auto device = aix::createDevice(deviceType);
    if (!device)
    {
        std::cerr << "Currently only Apple Silicon hardware acceleration is supported." << std::endl;
        exit(-1);
    }

    // Create a GPT2 model.
    auto model = GPT2(hParams["nVocab"], hParams["nCtx"], hParams["nEmbd"], hParams["nHeads"], hParams["nLayers"]);

    // Load the GPT2 model weights published by OpenAI.
    aix::load(model, modelFile);

    model.to(device);

    std::cout << "Prompt: " << prompt << std::endl;

    // Create the initial token ids for the prompt.
    auto inputTokenIds = bpe.encode(prompt);

    // Auto-regressive decoding loop: generate/predict the next token and append it to the initial tokens to predict
    // the following token.
    for (size_t i=0; i<hParams["nCtx"]; ++i)
    {
        // The GPT-2 model was not trained with start-of-sentence (SOS) or end-of-sentence (EOS) tokens.
        // Therefore, we can't determine when to stop generating the next token. Thus, we generate a specific number
        // of tokens, ensuring it does not exceed the context length.
        if (inputTokenIds.size() >= hParams["nCtx"]) break;

        // Convert the token IDs into a tensor.
        auto inputs = aix::Tensor(inputTokenIds.data(), inputTokenIds.size(), aix::DataType::kInt64,
                                  aix::Shape{inputTokenIds.size()}).to(device);

        // Predict the next token (either a word or a sub-word).
        auto logits = model.forward(inputs);
        auto nextTokenTensor = aix::argmax(logits[-1]);     // Greedy sampling. Selecting the highest prob token.

        // Synchronize to read data on the CPU.
        device->commitAndWait();

        // Decode the new token ID and print it.
        auto nextTokenId = nextTokenTensor.value().item<int32_t>();     // Argmax return type is always int32_t.
        std::cout << bpe.decode({nextTokenId}) << std::flush;

        // Append the new token ID to the current token sequence to predict the following token in the next iteration.
        inputTokenIds.emplace_back(nextTokenId);
    }

    return 0;
}
