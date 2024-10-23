config_setting(
    name = "optimized_mode",
    define_values = {"compilation_mode": "optimized"}
    # bazel build //:first_assignment --verbose_failures \
    #                    --define compilation_mode=optimized
)

cc_binary(
    name = "first_assignment",
    srcs = glob([
        "include/pc/*.hpp",
        "src/*.cpp",
        "tests/*.cpp",
        "benchmarks/*.cpp",
        "fuzz/*.cpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@valfuzz//:libvalfuzz",
    ],
    copts = select({
        ":optimized_mode": [
                "-O3",
                "-ffast-math",
                "-march=native"
        ],
        "//conditions:default": ["-O0"]
    }) + [
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Iinclude",
        "-std=c++23",
    ],
)
