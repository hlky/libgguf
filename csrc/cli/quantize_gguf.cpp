#include "conversion/converter.h"
#include "cli/quantize/options.h"

#include <cstdio>
#include <exception>

int main(int argc, char ** argv) {
    try {
        cli_options options = parse_args(argc, argv);
        conversion_result result = convert(options);
        std::printf("Wrote %s\n", result.output_path.c_str());
        std::printf("Architecture: %s\n", result.arch.c_str());
        std::printf("File type: %s\n", result.file_type_name.c_str());
        std::printf("Tensor types: ");
        bool first = true;
        for (const auto & item : result.tensor_type_counts) {
            if (!first) {
                std::printf(", ");
            }
            first = false;
            std::printf("%s=%d", item.first.c_str(), item.second);
        }
        std::printf("\n");
        if (!result.fallback_counts.empty()) {
            std::printf("Fallbacks: ");
            first = true;
            for (const auto & item : result.fallback_counts) {
                if (!first) {
                    std::printf(", ");
                }
                first = false;
                std::printf("%s=%d", item.first.c_str(), item.second);
            }
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
}