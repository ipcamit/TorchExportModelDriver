//
// Created by amit on 2/22/25.
//

/*
 * Example yaml file for TorchExport
 * WARNING: This is highly experimental and subjucted to change as the driver
 *          and the export process become more mature.
 * cutoff:
 * n_layers:
 * device:
 * number_of_inputs:
 *
 * FUTURE OPTIONS:
 * platforms:
 * model_name:
 *  platform:
 *    device:
 *    name:
 */

#ifndef YAML_HPP
#define YAML_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

class YAMLReader {
public:
    using YAMLMap = std::map<std::string, std::string>;
    using YAMLNestedMap = std::map<std::string, YAMLMap>;

    YAMLReader() = default;

    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open YAML file: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            parseLine(line);
        }

        file.close();
        return true;
    }

    std::string get(const std::string& key) const {
        auto it = data.find(key);
        if (it != data.end()) {
            return it->second;
        }
        return "";
    }

    void printAll() const {
        for (const auto& [key, value] : data) {
            std::cout << key << ": " << value << std::endl;
        }
    }

private:
    YAMLMap data;

    void parseLine(const std::string& line) {
        std::string trimmed = trim(line);

        // Skip empty lines and comments
        if (trimmed.empty() || trimmed[0] == '#') return;

        size_t delimiterPos = trimmed.find(':');
        if (delimiterPos != std::string::npos) {
            std::string key = trim(trimmed.substr(0, delimiterPos));
            std::string value = trim(trimmed.substr(delimiterPos + 1));
            data[key] = value;
        }
    }

    [[nodiscard]] std::string trim(const std::string& str) const {
        const char* whitespace = " \t\n\r\f\v";
        size_t start = str.find_first_not_of(whitespace);
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(whitespace);
        return str.substr(start, end - start + 1);
    }
};

#endif // YAML_HPP
