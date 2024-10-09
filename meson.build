project('pc', 'cpp',
  version : '1.0.0',
  license : 'MIT',
  default_options: [
    'cpp_std=c++23',
    'build.cpp_std=c++23'
  ]
)
add_project_arguments(
  [
    '-Wall',
    '-Wextra',
    '-pedantic',
    '-Werror',
    '-Wconversion',
    '-Wshadow'
  ],
  language: 'cpp'
)

# dependencies

cmake = import('cmake')
cmake_opts = cmake.subproject_options()
cmake_opts.add_cmake_defines({
  'VALFUZZ_BUILD_SHARED': 'ON'
})
valfuzz_proj = cmake.subproject(
  'valfuzz',
  options: cmake_opts
)
valfuzz_dep = valfuzz_proj.dependency('valfuzz_shared')

# project

base_sources = files(
  'src/pc.cpp',
  'src/ilp.cpp',
  'src/pragma.cpp',
)

if get_option('PC_BUILD_OPTIMIZED_AGGRESSIVE')
  add_project_arguments('-O3', '-ffast-math', '-march=native', language: 'cpp')
elif get_option('PC_BUILD_OPTIMIZED')
  add_project_arguments('-O3', language: 'cpp')
endif

if get_option('PC_BUILD_SHARED')
  shared_library('pc',
    sources: base_sources,
    include_directories: [
      'include',
    ],
    dependencies: [
      valfuzz_dep
    ],
  )
endif
if get_option('PC_BUILD_STATIC')
  static_library('pc',
    sources: base_sources,
    include_directories: [
      'include',
    ],
    dependencies: [
      valfuzz_dep
    ],
  )
endif
if get_option('PC_BUILD_TESTS')
  test_sources = files(
    'tests/pc_test.cpp',
    'fuzz/pc_fuzz.cpp',
    'benchmarks/pc_benchmark.cpp',
    'benchmarks/array_operations_benchmark.cpp',
    'benchmarks/pragma_benchmark.cpp',
  )
  executable('tests',
    sources: [
      test_sources,
      base_sources
    ],
    include_directories: [
      'include',
    ],
    dependencies: [
      valfuzz_dep
    ],
  )
endif
