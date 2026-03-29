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
#include "ModelKVCache.hpp"
#include "Runner.hpp"
// External includes
// System includes
#include <cstdlib>
#include <iostream>


namespace gpt2
{

class RunnerKVCache : public Runner
{
public:
    void run(const RunnerConfig& config) override
    {
        BPE bpe(config.bpeMergeFile, config.bpeVocabFile);

        auto device = createDevice(config);
        if (!device)
        {
            std::cerr << "Device type is not supported." << std::endl;
            exit(-1);
        }

        auto model = kvcache::GPT2(config.nVocab, config.nCtx, config.nEmbd, config.nHeads, config.nLayers);

        aix::nn::load(model, config.modelFile);

        model.to(device);
        model.prepare(device.get());

        // Inference does not require gradient.
        for (auto & [name, param] : model.parameters())
        {
            param = param.requireGrad(false);
        }

        std::cout << "Prompt: " << config.prompt << std::endl;

        // Create the initial token ids for the prompt.
        auto inputTokenIds = bpe.encode(config.prompt);

        // Create a KV cache to avoid recomputing attention for past tokens during generation.
        auto cache = kvcache::KVCache(config.nLayers, config.nCtx, config.nEmbd, device.get());

        // Auto-regressive decoding loop with KV cache: generate/predict the next token and append it to the initial
        // tokens to predict the following token.
        auto maxTokensToGenerate = std::min(config.maxOutputToken, config.nCtx - inputTokenIds.size());
        for (size_t i=0; i<maxTokensToGenerate; ++i)
        {
            // The GPT-2 model was not trained with start-of-sentence (SOS) or end-of-sentence (EOS) tokens.
            // Therefore, we can't determine when to stop generating the next token. Thus, we generate a specific number
            // of tokens, ensuring it does not exceed the context length.
            if (inputTokenIds.size() >= config.nCtx) break;

            // On the first step, process all prompt tokens (prefill). On subsequent steps, only the last token (decode).
            aix::Tensor inputs;
            if (i == 0)
            {
                inputs = aix::Tensor(inputTokenIds.data(), inputTokenIds.size(), aix::DataType::kInt64,
                                     aix::Shape{inputTokenIds.size()}, aix::dtype(aix::DataType::kInt32)).to(device);
            }
            else
            {
                ssize_t lastToken = inputTokenIds.back();
                inputs = aix::Tensor(&lastToken, 1, aix::DataType::kInt64,
                                     aix::Shape{1}, aix::dtype(aix::DataType::kInt32)).to(device);
            }

            auto startPos = inputTokenIds.size() - inputs.shape()[0];

            // Predict the next token (either a word or a sub-word).
            auto logits = model.forward(inputs, cache, startPos);
            auto nextTokenTensor = aix::argmax(logits[-1]);     // Greedy sampling. Selecting the highest prob token.

            // Synchronize to read data on the CPU.
            device->synchronize();

            // Decode the new token ID and print it.
            auto nextTokenId = nextTokenTensor.value().item<int32_t>();     // Argmax return type is always int32_t.
            std::cout << bpe.decode({nextTokenId}) << std::flush;

            // Append the new token ID to the current token sequence to predict the following token in the next iteration.
            inputTokenIds.emplace_back(nextTokenId);
        }
    }
};

}   // namespace gpt2
