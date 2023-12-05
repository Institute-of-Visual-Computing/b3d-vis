#include <device_launch_parameters.h>
#include <optix_device.h>
#include <owl/owl.h>

#include "SharedStructs.h"
#include "owl/owl_device.h"
#include "owl/common/math/vec.h"

using namespace b3d::renderer::nano;
using namespace owl;

extern "C" __constant__ LaunchParams optixLaunchParams;

OPTIX_BOUNDS_PROGRAM(volumeBounds)
(const void *geometryData, owl::box3f &primitiveBounds, const int primitiveID)
{
	const auto &self = *static_cast<const NanoVdbVolume*>(geometryData);

	primitiveBounds = self.worldAabb;
}

OPTIX_RAYGEN_PROGRAM(rayGeneration)()
{
	const auto& self = owl::getProgramData<RayGenerationData>();
        
    const int eyeIdx = optixLaunchParams.outputSurfaceIndex;
	const auto& camera = self.camera;
    const auto pixelId = owl::getLaunchIndex();
     
    const auto screen = (vec2f(pixelId) + vec2f(.5f)) / vec2f(self.frameBufferSize);
	auto color = vec4f{};
	color.x = screen.x;
	color.y = screen.y;
	color.z = 0.0f;
	color.w = 1.0;
	surf2Dwrite(owl::make_rgba(color),self.frameBufferPtr[0]/* self.outputSurfaceArray[eyeIdx]*/, sizeof(uint32_t) * pixelId.x, pixelId.y);
}
