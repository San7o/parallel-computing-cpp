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

## valfuzz
valfuzz_opts = cmake.subproject_options()
valfuzz_opts.add_cmake_defines({
  'VALFUZZ_BUILD_SHARED': 'ON',
  'VALFUZZ_BUILD_OPTIMIZED': get_option('PC_BUILD_OPTIMIZED'),
  'VALFUZZ_BUILD_OPTIMIZED_AGGRESSIVE': get_option('PC_BUILD_OPTIMIZED_AGGRESSIVE'),
})
valfuzz_proj = cmake.subproject(
  'valfuzz',
  options: valfuzz_opts
)
valfuzz_dep = valfuzz_proj.dependency('valfuzz_shared')

## tenno
tenno_opts = cmake.subproject_options()
tenno_opts.add_cmake_defines({
  'TENNO_BUILD_SHARED': 'ON',
  'TENNO_BUILD_OPTIMIZED': get_option('PC_BUILD_OPTIMIZED'),
  'TENNO_BUILD_OPTIMIZED_AGGRESSIVE': get_option('PC_BUILD_OPTIMIZED_AGGRESSIVE'),
})
tenno_proj = cmake.subproject(
  'tenno',
  options: tenno_opts
) 
tenno_dep = tenno_proj.dependency('tenno_shared')

# project

base_sources = files(
  'src/transpose.cpp',
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
      valfuzz_dep,
      tenno_dep,
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
      valfuzz_dep,
      tenno_dep,
    ],
  )
endif
if get_option('PC_BUILD_TESTS')
  test_sources = files(
    'tests/transpose_test.cpp',
    'fuzz/transpose_fuzz.cpp',
    'benchmarks/transpose_benchmark.cpp',
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
      valfuzz_dep,
      tenno_dep,
    ],
  )
endif
