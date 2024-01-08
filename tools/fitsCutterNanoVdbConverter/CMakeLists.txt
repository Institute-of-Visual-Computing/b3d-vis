
find_package(unofficial-cfitsio CONFIG REQUIRED)

add_executable(NanoCutter)
target_sources(NanoCutter PRIVATE main.cpp)
target_link_libraries(NanoCutter PRIVATE cfitsio)
target_compile_features(NanoCutter PUBLIC cxx_std_20)
