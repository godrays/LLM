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
// External includes
#include <aix.hpp>
// System includes
#include <string>


namespace gpt2
{

struct RunnerConfig
{
    std::string prompt;
    std::string modelFile;
    std::string bpeMergeFile;
    std::string bpeVocabFile;
    aix::DeviceType deviceType{aix::DeviceType::kCPU};
    size_t maxOutputToken{1024};
    size_t nVocab;
    size_t nCtx;
    size_t nEmbd;
    size_t nHeads;
    size_t nLayers;
};


class Runner
{
public:
    virtual ~Runner() = default;
    virtual void run(const RunnerConfig &config) = 0;
};

}   // namespace gpt2
