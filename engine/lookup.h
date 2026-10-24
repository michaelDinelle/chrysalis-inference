#pragma once
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct Lookup {
    static constexpr int MIN = 16;
    std::vector<int> hist;
    std::unordered_map<uint64_t, int> grams;

    static uint64_t gram3(const int *p) {
        return ((uint64_t)(uint32_t)p[0] * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)(uint32_t)p[1] * 0xC2B2AE3D27D4EB4Full) ^ (uint32_t)p[2];
    }

    void push(int t) {
        hist.push_back(t);
        const size_t n = hist.size();
        if (n >= 4) grams[gram3(&hist[n - 4])] = (int)n - 2;
    }

    void clear() { hist.clear(); grams.clear(); }

    void set(const int *tokens, int n) {
        if (hist.size() > (size_t)n || !std::equal(hist.begin(), hist.end(), tokens)) {
            hist.clear();
            grams.clear();
            grams.reserve((size_t)n + 4096);
        }
        for (int i = (int)hist.size(); i < n; i++) push(tokens[i]);
    }

    int find(int max, int vocab, int *out) const {
        const int n = (int)hist.size();
        if (n < MIN + 1 || max < 1) return 0;
        const auto it = grams.find(gram3(&hist[n - 3]));
        if (it == grams.end()) return 0;
        const int j = it->second;
        int m = 0;
        while (m < MIN && j - m >= 0 && hist[j - m] == hist[n - 1 - m]) m++;
        if (m < MIN) return 0;
        int k = 0;
        while (k < max && j + 1 + k < n && (unsigned)hist[j + 1 + k] < (unsigned)vocab) { out[k] = hist[j + 1 + k]; k++; }
        return k;
    }
};
