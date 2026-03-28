//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "Runner.hpp"
#include "RunnerNaive.hpp"
#include "RunnerKVCache.hpp"
// External includes
#include <aix.hpp>
#include <aixDevices.hpp>
#include <docopt/docopt.h>
// System includes
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>


using namespace gpt2;

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
    std::string modelImpl{"kvcache"};
    std::string modelPath;
    aix::DeviceType deviceType{aix::DeviceType::kCPU};
    size_t maxOutputToken{1024};
};

CmdLineOptions processCommandLineArguments(int argc, const char* argv[])
{
    static constexpr char USAGE[] =
    R"(
    GPT2 - Copyright (c) 2024-Present, Arkin Terli. All rights reserved.

    Usage:
        GPT2 --prompt=<text> --model=<type> --model-path=<path> --device=<type> [--model-impl=<impl>] [--max-output-token=<count>]

    Example:
        GPT2 --prompt="What do you know about artificial intelligence?" --model=124M --model-path=Resources/GPT2 --device=MCS --max-output-token=42

    Options:
        --prompt=<text>            Your prompt to the GPT2.
        --model=<type>             Model type to use. Options: [124M | 355M | 774M | 1558M]
        --model-impl=<impl>        Model implementation. Options: [kvcache | naive] [default: kvcache]
        --model-path=<path>        Model directory path.
        --device=<type>            Device type to use. Options: [CPU | MCS]
                                   MCS: Metal Compute Shaders for Apple Silicon.
        --max-output-token=<count> Maximum number of tokens to generate. [default: 1024]
    )";

    CmdLineOptions options;

    try
    {
        std::vector<std::string>  baseArgs{ argv + 1, argv + argc };

        auto args = docopt::docopt(USAGE, {argv + 1, argv + argc}, false, "GPT2 0.1.0");

        options.prompt  = args["--prompt"].asString();
        auto modelType  = args["--model"].asString();
        auto modelImpl  = args["--model-impl"].asString();
        auto modelPath  = args["--model-path"].asString();
        auto deviceType = args["--device"].asString();
        options.maxOutputToken = args["--max-output-token"].asLong();

        if (options.prompt.empty()) throw std::invalid_argument("Prompt cannot be empty.");

        if (modelType == "124M")        options.modelType = ModelConfigType::OPENAI_124M;
        else if (modelType == "355M")   options.modelType = ModelConfigType::OPENAI_355M;
        else if (modelType == "774M")   options.modelType = ModelConfigType::OPENAI_774M;
        else if (modelType == "1558M")  options.modelType = ModelConfigType::OPENAI_1558M;
        else throw std::invalid_argument("Unknown model type: " + modelType);

        if (modelImpl == "naive")       options.modelImpl = "naive";
        else if (modelImpl == "kvcache") options.modelImpl = "kvcache";
        else throw std::invalid_argument("Unknown model implementation: " + modelImpl);

        if (!modelPath.empty()) options.modelPath = modelPath;
        else throw std::invalid_argument("Invalid model path: " + modelPath);

        if (deviceType == "CPU")        options.deviceType = aix::DeviceType::kCPU;
        else if (deviceType == "MCS")   options.deviceType = aix::DeviceType::kGPU_METAL;
        else throw std::invalid_argument("Unknown device type: " + deviceType);
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
    auto cmdLineOptions = processCommandLineArguments(argc, argv);

    std::vector<std::unordered_map<std::string, size_t>>  modelParams
    {
        { {"nVocab", 50257}, {"nCtx", 1024}, {"nEmbd",  768}, {"nHeads", 12}, {"nLayers", 12}, },   // 124M
        { {"nVocab", 50257}, {"nCtx", 1024}, {"nEmbd", 1024}, {"nHeads", 16}, {"nLayers", 24}, },   // 355M
        { {"nVocab", 50257}, {"nCtx", 1024}, {"nEmbd", 1280}, {"nHeads", 20}, {"nLayers", 36}, },   // 774M
        { {"nVocab", 50257}, {"nCtx", 1024}, {"nEmbd", 1600}, {"nHeads", 25}, {"nLayers", 48}, },   // 1558M
    };

    std::vector<std::string> modelWeightsFilenames =
    {
        "oaiWeights124M.bin",
        "oaiWeights355M.bin",
        "oaiWeights774M.bin",
        "oaiWeights1558M.bin",
    };

    auto modelType    = static_cast<size_t>(cmdLineOptions.modelType);
    auto hParams      = modelParams[modelType];
    auto modelFile    = cmdLineOptions.modelPath + "/" + modelWeightsFilenames[modelType];
    auto bpeMergeFile = cmdLineOptions.modelPath + "/oaiBPEMergeRules.txt";
    auto bpeVocabFile = cmdLineOptions.modelPath + "/oaiBPEVocabs.txt";

    validateFileExistence(bpeMergeFile);
    validateFileExistence(bpeVocabFile);
    validateFileExistence(modelFile);

    RunnerConfig config
    {
        .prompt         = cmdLineOptions.prompt,
        .modelFile      = modelFile,
        .bpeMergeFile   = bpeMergeFile,
        .bpeVocabFile   = bpeVocabFile,
        .deviceType     = cmdLineOptions.deviceType,
        .maxOutputToken = cmdLineOptions.maxOutputToken,
        .nVocab         = hParams["nVocab"],
        .nCtx           = hParams["nCtx"],
        .nEmbd          = hParams["nEmbd"],
        .nHeads         = hParams["nHeads"],
        .nLayers        = hParams["nLayers"],
    };

    std::unique_ptr<Runner> runner;
    if (cmdLineOptions.modelImpl == "naive")        runner = std::make_unique<RunnerNaive>();
    else if (cmdLineOptions.modelImpl == "kvcache") runner = std::make_unique<RunnerKVCache>();

    runner->run(config);

    return 0;
}
