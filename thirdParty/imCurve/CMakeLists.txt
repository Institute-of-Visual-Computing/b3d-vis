find_package(imgui CONFIG REQUIRED)

add_library(imcurve_modefied STATIC)
target_sources(imcurve_modefied PUBLIC
	Curve.h
	Curve.cpp
)

target_include_directories(imcurve_modefied PUBLIC /)

target_compile_features(imcurve_modefied PUBLIC cxx_std_23)
target_link_libraries(imcurve_modefied 
  PRIVATE
	imgui::imgui
)

set_property(TARGET imcurve_modefied PROPERTY FOLDER "ThirdParty/imcurve_modefied")
