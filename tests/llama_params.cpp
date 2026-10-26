#include "server-schema.h"
#include <iostream>
#include <string>

int main() {
    common_params base;
    std::string line;
    if (!std::getline(std::cin, line)) return 1;
    const json d = json::parse(line);
    base.sampling.temp = d.at("temperature").get<float>();
    base.sampling.top_p = d.at("top_p").get<float>();
    base.sampling.top_k = d.at("top_k").get<int>();
    base.sampling.min_p = d.at("min_p").get<float>();
    base.sampling.penalty_present = d.at("presence_penalty").get<float>();
    base.sampling.dry_penalty_last_n = d.at("dry_penalty_last_n").get<int>();
    const std::vector<llama_logit_bias> eog;
    while (std::getline(std::cin, line)) {
        json out;
        try {
            out = server_schema::eval_llama_cmpl_schema(nullptr, base, eog, json::parse(line)).to_json();
        } catch (const std::exception &e) {
            out = {{"error", e.what()}};
        }
        std::cout << out.dump() << std::endl;
    }
    return 0;
}
