//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

// Project includes
#include "BPE.hpp"
#include "ModelNaive.hpp"
#include "Runner.hpp"
// External includes
// System includes
#include <chrono>
#include <cstdlib>
#include <iostream>


namespace gpt2
{

class RunnerNaive : public Runner
{
public:
    RunMetrics run(const RunnerConfig &config) override
    {
        aix::NoGradGuard guard;
        RunMetrics metrics;

        BPE bpe(config.bpeMergeFile, config.bpeVocabFile);

        auto device = createDevice(config);
        if (!device)
        {
            std::cerr << "Device type is not supported." << std::endl;
            exit(-1);
        }

        auto model = naive::GPT2(config.nVocab, config.nCtx, config.nEmbd, config.nHeads, config.nLayers);

        aix::nn::load(model, config.modelFile);

        model.to(device);
        device->synchronize();

        for (auto & [name, param] : model.parameters())
        {
            param = param.requireGrad(false);
        }

        std::cout << "Prompt: " << config.prompt << std::endl;

        auto inputTokenIds = bpe.encode(config.prompt);
        metrics.promptTokenCount = inputTokenIds.size();
        auto maxTokensToGenerate = std::min(config.maxOutputToken, config.nCtx - inputTokenIds.size());
        auto generationStart = std::chrono::steady_clock::now();

        for (size_t i = 0; i < maxTokensToGenerate; ++i)
        {
            if (inputTokenIds.size() >= config.nCtx) break;

            auto stepStart = std::chrono::steady_clock::now();

            auto inputs = aix::Tensor(inputTokenIds.data(), inputTokenIds.size(), aix::DataType::kInt64,
                                      aix::Shape{inputTokenIds.size()}, aix::dtype(aix::DataType::kInt32)).to(device);

            auto logits = model.forward(inputs);
            auto nextTokenTensor = aix::argmax(logits[-1]);

            device->synchronize();

            auto stepEnd = std::chrono::steady_clock::now();
            auto stepDurationMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
            if (i == 0) metrics.prefillDurationMs += stepDurationMs;
            else metrics.decodeDurationMs += stepDurationMs;

            auto nextTokenId = nextTokenTensor.value().item<int32_t>();
            std::cout << bpe.decode({nextTokenId}) << std::flush;

            inputTokenIds.emplace_back(nextTokenId);
            ++metrics.generatedTokenCount;
        }

        auto generationEnd = std::chrono::steady_clock::now();
        metrics.generationDurationMs = std::chrono::duration<double, std::milli>(generationEnd - generationStart).count();

        return metrics;
    }
};

}   // namespace gpt2
