#include "conversion/native_safetensors_writer.h"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <filesystem>

template<class Character>
int run_cli(int argc, Character** argv) {
    try {
        const auto argument=[](const Character* value) {
            const auto utf8=std::filesystem::path(value).u8string();
            return std::string(utf8.begin(),utf8.end());
        };
        std::string source, destination, name;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::runtime_error("option requires a value");
            const std::string key = argument(argv[i]);
            if (key == "--src") source = argument(argv[i + 1]);
            else if (key == "--dst") destination = argument(argv[i + 1]);
            else if (key == "--format") name = argument(argv[i + 1]);
            else throw std::runtime_error("unknown option " + key);
        }
        if (source.empty() || destination.empty() || name.empty())
            throw std::runtime_error("usage: libgguf_quantize_native --src input.safetensors --dst output.safetensors --format S4G64");
        const std::string names[] = {"S4G32", "S4G64", "S8G32", "S4H256"};
        int format = -1;
        for (int i = 0; i < 4; ++i) if (name == names[i]) format = i;
        if (format < 0) throw std::runtime_error("unknown native format");
        convert_native_safetensors(source, destination, static_cast<libgguf_native_format>(format));
        std::printf("Wrote %s (%s)\n", destination.c_str(), name.c_str());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
    return 0;
}

#if defined(_WIN32)
int wmain(int argc,wchar_t** argv) {return run_cli(argc,argv);}
#else
int main(int argc,char** argv) {return run_cli(argc,argv);}
#endif
