cc_library(
    name = "pc",
    srcs = [
        "pc.cpp",
        "ilp.cpp",
        "pragma.cpp",
    ],
    hdrs = [
        "//include/pc:pc.hpp",
        "//include/pc:ilp.hpp",
        "//include/pc:pragma.hpp",
    ],  # Refer to the header in the `include/pc` directory
    includes = ["//include"],  # Add the base include directory to the include path
    copts = ["-Iinclude"],
    visibility = ["//visibility:public"],  # Allow the main package to depend on this library
)
