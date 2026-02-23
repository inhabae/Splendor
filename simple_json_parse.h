#pragma once

#include <string>

namespace simple_json {

inline int extractInt(const std::string& json, const std::string& key, int default_value = 0) {
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return default_value;
    size_t colon = json.find(":", key_pos);
    if (colon == std::string::npos) return default_value;
    size_t end = json.find_first_of(",}", colon + 1);
    if (end == std::string::npos) end = json.size();
    std::string val = json.substr(colon + 1, end - colon - 1);
    val.erase(0, val.find_first_not_of(" \t\n\r"));
    size_t last = val.find_last_not_of(" \t\n\r");
    if (last == std::string::npos) return default_value;
    val.erase(last + 1);
    try { return std::stoi(val); } catch (...) { return default_value; }
}

inline std::string extractStr(const std::string& json, const std::string& key) {
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return "";
    size_t colon = json.find(":", key_pos);
    if (colon == std::string::npos) return "";
    size_t q1 = json.find("\"", colon + 1);
    size_t q2 = (q1 == std::string::npos) ? std::string::npos : json.find("\"", q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) return "";
    return json.substr(q1 + 1, q2 - q1 - 1);
}

inline std::string extractObject(const std::string& json, const std::string& key) {
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return "{}";
    size_t brace = json.find("{", key_pos);
    if (brace == std::string::npos) return "{}";

    int depth = 1;
    size_t cur = brace + 1;
    while (cur < json.size() && depth > 0) {
        if (json[cur] == '{') depth++;
        else if (json[cur] == '}') depth--;
        cur++;
    }
    return json.substr(brace, cur - brace);
}

} // namespace simple_json
