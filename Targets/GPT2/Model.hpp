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


class GeLU : public aix::nn::Module
{
public:
    // Constructor.
    GeLU() = default;

    aix::Tensor forward(aix::Tensor x) const override
    {
        return 0.5 * x * (1.0 + aix::tanh(std::sqrtf(2.0 / std::numbers::pi) * (x + 0.044715 * x.pow(3))));
    }
};


class Softmax : public aix::nn::Module
{
public:
    // Constructor.
    Softmax() = default;

    // Constructor.
    explicit Softmax(ssize_t dim=0, bool keepDim=false) : m_dim{dim}, m_keepDim{keepDim} { }

    aix::Tensor forward(aix::Tensor x) const override
    {
        x = (x - x.max(m_dim, m_keepDim)).exp();
        return x / x.sum(m_dim, m_keepDim);
    }

private:
    ssize_t m_dim{0};
    bool m_keepDim{false};
};


class Linear : public aix::nn::Module
{
public:
    // Constructor.
    Linear() = default;

    // Constructor.
    explicit Linear(size_t numInputs, size_t numOutputs)
    {
        m_w = aix::randn({numInputs, numOutputs}, aix::requireGrad(true));
        m_b = aix::randn({1,         numOutputs}, aix::requireGrad(true));

        // Register learnable parameters.
        registerParameter("b", m_b);
        registerParameter("w", m_w);
    }

    aix::Tensor forward(aix::Tensor x) const override
    {
        return matmul(x, m_w) + m_b;
    }

private:
    aix::Tensor  m_w;
    aix::Tensor  m_b;
};


class LayerNorm : public aix::nn::Module
{
public:
    // Constructor.
    LayerNorm() = default;

    // Constructor.
    explicit LayerNorm(size_t gSize, size_t bSize, float eps=1e-5, ssize_t dim=0, bool keepDim=false)
        : m_eps{eps}, m_dim{dim}, m_keepDim{keepDim}
    {
        m_b = aix::randn({bSize}, aix::requireGrad(true));
        m_g = aix::randn({gSize}, aix::requireGrad(true));

        registerParameter("b", m_b);
        registerParameter("g", m_g);
    }

    aix::Tensor forward(aix::Tensor x) const override
    {
        auto mean = x.mean(m_dim, m_keepDim);
        auto variance = x.var(m_dim, true, m_keepDim);
        x = (x - mean) / (variance + m_eps).sqrt();
        return m_g * x + m_b;
    }

private:
    aix::Tensor  m_b;
    aix::Tensor  m_g;
    float   m_eps{1e-5};
    ssize_t m_dim{0};
    bool    m_keepDim{false};
};


class Embeddings : public aix::nn::Module
{
public:
    // Constructor.
    Embeddings() = default;

    // Constructor.
    explicit Embeddings(size_t numInputs, size_t numOutputs)
    {
        m_w = aix::randn({numInputs, numOutputs}, aix::requireGrad(true));

        registerParameter("w", m_w);
    }

    aix::Tensor forward(aix::Tensor inputTokenIds) const override
    {
        return m_w.indexSelect(0, inputTokenIds);
    }

    aix::Tensor matmulWithTranspose(const aix::Tensor& x) const
    {
        return x.matmul(m_w.transpose(0, 1));
    }

private:
    aix::Tensor  m_w;
};


class FeedForwardNet : public aix::nn::Module
{
public:
    // Constructor.
    FeedForwardNet() = default;

    // Constructor.
    explicit FeedForwardNet(size_t emdSize)
    {
        m_fc    = Linear(emdSize, emdSize * 4);
        m_cProj = Linear(emdSize * 4, emdSize);

        registerModule(m_fc);
        registerModule(m_cProj);
    }

    aix::Tensor forward(aix::Tensor x) const override
    {
        // Project up.
        auto a = GeLU().forward(m_fc.forward(x));       // {seq, embd} --> {seq, 4*embd}
        // Project back down.
        return m_cProj.forward(a);                      // {seq, 4*embd} --> {seq, embd}
    }

private:
    Linear  m_fc;
    Linear  m_cProj;
};


class MultiHeadAttention : public aix::nn::Module
{
public:
    // Constructor.
    MultiHeadAttention() = default;

    // Constructor.
    explicit MultiHeadAttention(size_t embdDim, size_t numHeads) : m_embdDim{embdDim}, m_numHeads{numHeads}
    {
        if (embdDim % numHeads != 0)
        {
            throw std::invalid_argument("Embedding size must be multiple of number of heads.");
        }

        m_cAtt  = Linear(embdDim, embdDim * 3);
        m_cProj = Linear(embdDim, embdDim);

        registerModule(m_cAtt );
        registerModule(m_cProj);
    }

    aix::Tensor forward(aix::Tensor x) const override
    {
        // QKV projection.
        x = m_cAtt.forward(x);                  // {seq, embd} --> {seq, 3*embd}

        // Causal mask to hide future inputs from being attended to. ( mask{seq, seq} )
        auto causalMask = aix::ones({x.shape()[0], x.shape()[0]}, aix::device(x.device())).triu(1) * -1e10;

        // Split into {Q, K, V}
        auto qkv = x.split(m_embdDim, -1);     // {seq, 3*embd} --> {3, seq, embd}

        // Split each in qkv into n heads/chucks.
        auto qHeads = qkv[0].split(m_embdDim / m_numHeads, -1);     // Q --> n heads.
        auto kHeads = qkv[1].split(m_embdDim / m_numHeads, -1);     // K --> n heads.
        auto vHeads = qkv[2].split(m_embdDim / m_numHeads, -1);     // V --> n heads.

        // [3, heads, seq, embd/heads] --> [heads, seq, embd/heads]
        std::vector<aix::Tensor> outHeads;
        for (size_t i=0; i<qHeads.size(); ++i)
        {
            outHeads.emplace_back(attention(qHeads[i], kHeads[i], vHeads[i], causalMask));
        }

        // Merge heads.
        x = aix::hstack(outHeads);          // [heads, seq, embd/heads] --> {seq, embd}

        // Out projection.
        return m_cProj.forward(x);          // {seq, embd} --> {seq, embd}
    }

private:
    static aix::Tensor attention(const aix::Tensor& q, const aix::Tensor& k, const aix::Tensor& v,
                                 const aix::Tensor& mask)
    {
        return Softmax(-1, true).forward(q.matmul(k.transpose(0, 1)) / std::sqrt(q.shape().back()) + mask).matmul(v);
    }

    size_t  m_embdDim{0};
    size_t  m_numHeads{0};
    Linear  m_cAtt;
    Linear  m_cProj;
};


class TransformerBlock : public aix::nn::Module
{
public:
    // Constructor.
    TransformerBlock() = default;

    // Constructor.
    TransformerBlock(size_t embdDim, size_t numHeads)
    {
        m_mha = MultiHeadAttention(embdDim, numHeads);
        m_ln1 = LayerNorm(embdDim, embdDim, 1e-5, -1, true);
        m_ln2 = LayerNorm(embdDim, embdDim, 1e-5, -1, true);
        m_ffn = FeedForwardNet(embdDim);

        registerModule(m_mha);
        registerModule(m_ln1);
        registerModule(m_ln2);
        registerModule(m_ffn);
    }

    aix::Tensor forward(aix::Tensor x) const override
    {
        // Multi-head causal self-attention.
        x = x + m_mha.forward(m_ln1.forward(x));        // {seq, embd} --> {seq, embd}

        // Position-wise feed-forward network.
        return x + m_ffn.forward(m_ln2.forward(x));     // {seq, embd} --> {seq, embd}
    }

private:
    MultiHeadAttention  m_mha;
    LayerNorm m_ln1;
    LayerNorm m_ln2;
    FeedForwardNet m_ffn;
};


class GPT2 : public aix::nn::Module
{
public:
    // Constructor.
    GPT2() = default;

    // Constructor.
    explicit GPT2(size_t vocabSize, size_t ctxSize, size_t embdDim, size_t numHeads, size_t numLayers)
        : m_numLayers{numLayers}
    {
        // The architecture uses only the decoder stack of the original transformer model.
        m_wte = Embeddings(vocabSize, embdDim);
        m_wpe = Embeddings(ctxSize, embdDim);
        m_layerNorm = LayerNorm(embdDim, embdDim, 1e-5, -1, true);

        for (size_t i=0; i<numLayers; ++i)
        {
            m_transformerBlocks.emplace_back(embdDim, numHeads);
            registerModule(m_transformerBlocks.back());
        }

        registerModule(m_layerNorm);
        registerModule(m_wpe);
        registerModule(m_wte);
    }

    aix::Tensor forward(aix::Tensor inputs) const override
    {
        // Text and positional embeddings.
        auto range = aix::arange(0, inputs.shape()[0], 1, aix::dtype(aix::DataType::kInt32).device(inputs.device()));
        auto x = m_wte.forward(inputs) + m_wpe.forward(range);

        // Transformer decoder stack.
        for (size_t i=0; i<m_numLayers; ++i)
        {
            x = m_transformerBlocks[i].forward(x);
        }

        // Projection to vocabulary. The final layer normalization is specific to the GPT2 architecture.
        // It is not present in the original GPT and Transformer papers.
        // NOTE: Softmax is not applied at the end, so the outputs will be logits instead of probabilities.
        return m_wte.matmulWithTranspose(m_layerNorm.forward(x));
    }

private:
    size_t      m_numLayers{0};
    Embeddings  m_wpe;
    Embeddings  m_wte;
    LayerNorm   m_layerNorm;
    std::vector<TransformerBlock>  m_transformerBlocks;
};
