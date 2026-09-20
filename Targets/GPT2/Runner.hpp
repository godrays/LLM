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
#include <aixDevices.hpp>
// System includes
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>


namespace gpt2
{

struct RunMetrics
{
    size_t promptTokenCount{0};
    size_t generatedTokenCount{0};
    double totalDurationMs{0.0};
    double generationDurationMs{0.0};
    double prefillDurationMs{0.0};
    double decodeDurationMs{0.0};
};

struct RunnerConfig
{
    std::string prompt;
    std::string modelFile;
    std::string bpeMergeFile;
    std::string bpeVocabFile;
    aix::DeviceType deviceType{aix::DeviceType::kCPU};
    bool autoDevice{true};
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
    virtual RunMetrics run(const RunnerConfig &config) = 0;

    static void printRunMetrics(const RunMetrics& metrics)
    {
        auto totalDurationSec = metrics.totalDurationMs / 1000.0;
        auto generationDurationSec = metrics.generationDurationMs / 1000.0;
        auto tokensPerSecond = generationDurationSec > 0.0
                             ? static_cast<double>(metrics.generatedTokenCount) / generationDurationSec
                             : 0.0;

        std::cout << "\n\n"
                  << std::fixed << std::setprecision(2)
                  << "Total duration: " << metrics.totalDurationMs << " ms (" << totalDurationSec << " s)\n"
                  << "Generation duration: " << metrics.generationDurationMs << " ms (" << generationDurationSec << " s)\n"
                  << "Prefill duration: " << metrics.prefillDurationMs << " ms\n"
                  << "Tokens per second: " << tokensPerSecond << " tps\n";
    }

protected:
    static std::unique_ptr<aix::Device> createDevice(const RunnerConfig &config)
    {
        if (config.autoDevice)
        {
            auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
            if (device)
            {
                std::cout << "Auto-selected device: " << device->name() << std::endl;
                return device;
            }
            device = aix::createDevice(aix::DeviceType::kCPU);
            if (device)
            {
                std::cout << "Auto-selected device: " << device->name() << std::endl;
                return device;
            }
            std::cerr << "No supported device found." << std::endl;
            return nullptr;
        }
        return aix::createDevice(config.deviceType);
    }
};

}   // namespace gpt2
