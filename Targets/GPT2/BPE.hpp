//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


class BPE
{
public:
    // Constructor
    BPE() = default;

    // Constructor
    BPE(const std::string& mergesFile, const std::string& vocabsFile)
        : m_re(L"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\\d+| ?[^\\s\\w]+|\\s+)")
    {
        m_bpeRanks.clear();
        m_b2u.clear();
        m_u2b.clear();
        m_t2i.clear();
        m_i2t.clear();

        loadMergeRules(mergesFile);
        loadVocab(vocabsFile);
        bytesToUnicode();
    }

    std::vector<ssize_t> encode(const std::string& text, const std::string& eot="<|endoftext|>")
    {
        std::vector<std::string> tokens;
        size_t s = 0;
        size_t i = text.find(eot);
        while (i != std::string::npos)
        {
            tokenize(text.substr(s, i - s), tokens);
            tokens.emplace_back(eot);
            s = i + eot.size();
            i = text.find(eot, s);
        }
        tokenize(text.substr(s), tokens);

        std::vector<ssize_t> tokenIds;
        tokenIds.reserve(tokens.size());
        for (const auto& token : tokens)
        {
            tokenIds.emplace_back(m_t2i.at(token));
        }
        return tokenIds;
    }

    std::string decode(const std::vector<ssize_t>& tokenIds)
    {
        std::string str;
        for (ssize_t id : tokenIds)
        {
            str += m_i2t.at(id);
        }

        auto wstr = utf8ToWString(str);
        std::string text;
        for (wchar_t c : wstr)
        {
            text.push_back(char(m_u2b.at(c)));
        }
        return text;
    }

private:
    // pairWStringHasher is used in BPERanks to make a pair of wstrings hashable, so the pair can be used as the key to
    // unordered_map.
    struct pairWStringHasher
    {
        size_t operator()(const std::pair<std::wstring, std::wstring>& p) const
        {
            auto hash1 = std::hash<std::wstring>{}(p.first);
            auto hash2 = std::hash<std::wstring>{}(p.second);
            return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
        }
    };

    // BPERanks maps each merge rule, which is a pair of wstrings, to its rank. This mapping allows quick lookup for
    // the optimal merge rule.
    using BPERanks = std::unordered_map<std::pair<std::wstring, std::wstring>, ssize_t, pairWStringHasher>;

    static std::wstring utf8ToWString(const std::string& str)
    {
        std::wstring wstr;
        wstr.reserve(str.size());   // Reserve enough space in advance.

        size_t i = 0;
        while (i < str.size())
        {
            wchar_t wc;
            auto ch = static_cast<unsigned char>(str[i]);

            if (ch <= 0x7f)
            {
                wc = ch;
                i += 1;
            }
            else if (ch <= 0xdf)
            {
                wc = (ch & 0x1f) << 6;
                wc |= (static_cast<unsigned char>(str[i + 1]) & 0x3f);
                i += 2;
            }
            else if (ch <= 0xef)
            {
                wc = (ch & 0x0f) << 12;
                wc |= (static_cast<unsigned char>(str[i + 1]) & 0x3f) << 6;
                wc |= (static_cast<unsigned char>(str[i + 2]) & 0x3f);
                i += 3;
            }
            else if (ch <= 0xf7)
            {
                wc = (ch & 0x07) << 18;
                wc |= (static_cast<unsigned char>(str[i + 1]) & 0x3f) << 12;
                wc |= (static_cast<unsigned char>(str[i + 2]) & 0x3f) << 6;
                wc |= (static_cast<unsigned char>(str[i + 3]) & 0x3f);
                i += 4;
            }
            else
            {
                // Invalid UTF-8 leading byte.
                throw std::runtime_error("Invalid UTF-8 sequence.");
            }

            wstr.push_back(wc);
        }
        return wstr;
    }

    static std::string wstringToUTF8(const std::wstring& wstr)
    {
        if (wstr.empty()) return {};

        std::string str;
        str.reserve(wstr.size() * 4);   // Reserve enough space in advance.

        for (const auto wc : wstr)
        {
            if (wc <= 0x7f)
            {
                str.push_back(static_cast<char>(wc));
            }
            else if (wc <= 0x7ff)
            {
                str.push_back(static_cast<char>(0xc0 | ((wc >> 6) & 0x1f)));
                str.push_back(static_cast<char>(0x80 | (wc & 0x3f)));
            }
            else if (wc <= 0xffff)
            {
                str.push_back(static_cast<char>(0xe0 | ((wc >> 12) & 0x0f)));
                str.push_back(static_cast<char>(0x80 | ((wc >> 6) & 0x3f)));
                str.push_back(static_cast<char>(0x80 | (wc & 0x3f)));
            }
            else if (wc <= 0x10ffff)
            {
                str.push_back(static_cast<char>(0xf0 | ((wc >> 18) & 0x07)));
                str.push_back(static_cast<char>(0x80 | ((wc >> 12) & 0x3f)));
                str.push_back(static_cast<char>(0x80 | ((wc >> 6) & 0x3f)));
                str.push_back(static_cast<char>(0x80 | (wc & 0x3f)));
            }
        }
        return str;
    }

    void bytesToUnicode()
    {
        auto insertRange = [&](ssize_t start, ssize_t end)
        {
            for (ssize_t c = start; c <= end; c++)
            {
                m_b2u.insert({uint8_t(c), wchar_t(c)});
            }
        };

        m_b2u.clear();
        insertRange(L'!', L'~');
        insertRange(L'¡', L'¬');
        insertRange(L'®', L'ÿ');

        ssize_t n = 0;
        for (ssize_t b = 0; b < 256; b++)
        {
            if (!m_b2u.contains(uint8_t(b)))
            {
                m_b2u.insert({uint8_t(b), wchar_t(256 + n)});
                n++;
            }
        }

        m_u2b.clear();
        for (auto e : m_b2u)
        {
            m_u2b.insert({e.second, e.first});
        }
    }

    // Given a token as a UTF8 string, encode each byte into a wchar_t.
    void byteEncodeToken(const std::string& token, std::wstring& encodedToken) const
    {
        encodedToken.resize(0);
        for (char c : token)
        {
            wchar_t wc = m_b2u.at(uint8_t(c));
            encodedToken.push_back(wc);
        }
    }

    void loadMergeRules(const std::string& filename)
    {
        std::fstream ins(filename, std::ios::in);
        m_bpeRanks.clear();

        std::string line;
        ssize_t n = 0;
        while (std::getline(ins, line))
        {
            if (n++ > 0)        // Skip the version comment.
            {
                auto d = line.find(' ');        // Merges file uses ASCII spaces.
                m_bpeRanks.insert({{utf8ToWString(line.substr(0, d)), utf8ToWString(line.substr(d + 1))}, n - 1});
            }
        }

        ins.close();
    }

    static void getPairs(const std::wstring& word, std::vector<std::pair<std::wstring, std::wstring>>& pairs)
    {
        pairs.clear();
        if (word.size() < 2) return;
        pairs.reserve(word.size());

        wchar_t previous = word[0];
        for (size_t i=1; i<word.size(); ++i)
        {
            pairs.emplace_back(std::wstring(1, previous), std::wstring(1, word[i]));
            previous = word[i];
        }
    }

    void bpe(const std::wstring& token, std::vector<std::wstring>* result)
    {
        std::set<ssize_t> merged;       // Stores indices in pairs that were merged.
        auto left = [](ssize_t i, std::set<ssize_t>& merged) -> ssize_t
        {
            for (ssize_t j = i - 1; j >= -1; j--)
            {
                if (!merged.contains(j)) return j;
            }
            return -1;
        };

        auto right = [](ssize_t i, ssize_t cap, std::set<ssize_t>& merged) -> ssize_t
        {
            for (ssize_t j = i + 1; j < cap; j++)
            {
                if (!merged.contains(j)) return j;
            }
            return cap;
        };

        std::vector<std::pair<std::wstring, std::wstring>> pairs;
        getPairs(token, pairs);

        while (true)
        {
            ssize_t minScore = std::numeric_limits<ssize_t>::max();
            ssize_t toMerge  = -1;      // Indices into pairs.

            for (ssize_t i=0; i<static_cast<ssize_t>(pairs.size()); ++i)
            {
                if (!merged.contains(i))         // Pair i is not merged.
                {
                    auto iter = m_bpeRanks.find(pairs[i]);
                    ssize_t score = iter != m_bpeRanks.end() ? iter->second : std::numeric_limits<ssize_t>::max();
                    if (score < minScore)
                    {
                        minScore = score;
                        toMerge = i;
                    }
                }
            }

            if (toMerge == -1) break;

            merged.insert(toMerge);
            auto mergeInto = pairs[toMerge].first + pairs[toMerge].second;

            ssize_t l = left(toMerge, merged);
            if (l >= 0) pairs[l].second = mergeInto;
            ssize_t r = right(toMerge, static_cast<ssize_t>(pairs.size()), merged);
            if (r < static_cast<ssize_t>(pairs.size())) pairs[r].first = mergeInto;
        }

        if (merged.size() == pairs.size())
        {
            result->emplace_back(token);
        }
        else
        {
            for (ssize_t i = 0; i < static_cast<ssize_t>(pairs.size()); ++i)
            {
                if (!merged.contains(i))
                {
                    if (left(i, merged) < 0) result->emplace_back(pairs[i].first);
                    result->emplace_back(pairs[i].second);
                }
            }
        }
    }

    void tokenize(const std::string& text, std::vector<std::string>& result)
    {
        auto wtext = utf8ToWString(text);
        std::wsmatch match;
        std::wstring::const_iterator searchStart(wtext.cbegin());

        while (std::regex_search(searchStart, wtext.cend(), match, m_re))
        {
            auto wtoken = match.str();
            std::wstring encodedToken;
            byteEncodeToken(wstringToUTF8(wtoken), encodedToken);

            std::vector<std::wstring> bpeTokens;
            bpe(encodedToken, &bpeTokens);

            for (const auto& ws : bpeTokens)
            {
                result.emplace_back(wstringToUTF8(ws));
            }

            searchStart = match.suffix().first;
        }
    }

    void loadVocab(const std::string& filename)
    {
        std::fstream ins(filename, std::ios::in);

        m_t2i.clear();
        m_i2t.clear();

        std::string line;
        std::string token;
        ssize_t n = 0;
        while (std::getline(ins, line))
        {
            if (n % 2 == 0)
            {
                token = line;
            }
            else
            {
                m_t2i.insert({token, std::stoi(line)});
                m_i2t.insert({std::stoi(line), token});
            }
            n++;
        }

        ins.close();
    }

    std::wregex m_re;
    BPERanks m_bpeRanks;
    std::unordered_map<uint8_t, wchar_t>      m_b2u;
    std::unordered_map<wchar_t, uint8_t>      m_u2b;
    std::unordered_map<std::string, ssize_t>  m_t2i;
    std::unordered_map<ssize_t, std::string>  m_i2t;
};
