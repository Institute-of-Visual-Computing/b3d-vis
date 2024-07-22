#include "NanoRenderer.h"
#include <nanovdb/NanoVDB.h>
#include <owl/helper/cuda.h>
#include "cuda_runtime.h"

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/Primitives.h>

#include <nanovdb/util/IO.h>

#include <filesystem>


#include "SharedStructs.h"
#include "owl/owl_host.h"

#include <format>

#include "DebugDrawListBase.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <NanoCutterParser.h>

#include <OwlExtensions.h>

#include "FoveatedRendering.h"


#define FOVEATED


extern "C" char NanoRenderer_ptx[];
extern "C" uint8_t NanoRenderer_optixir[];
extern "C" uint32_t NanoRenderer_optixir_length;


using namespace b3d::renderer;
using namespace b3d::renderer::nano;
using namespace owl::common;


namespace
{
	struct GuiData
	{
		bool fillBox{ false };
		std::array<float, 3> fillColor{ 0.8f, 0.3f, 0.2f };

		std::array<float, 2> sampleRemapping{ 0.0f, 0.1f };

		std::array<float, 2> fovealPoint{ 0.0f,0.0f };

		SampleIntegrationMethod sampleIntegrationMethode{ SampleIntegrationMethod::maximumIntensityProjection };
	};

	GuiData guiData{};


	auto computeStableEpsilon(const float f) -> float
	{
		return abs(f) * static_cast<float>(1. / (1 << 21));
	}

	auto computeStableEpsilon(const vec3f& v) -> float
	{
		return max(max(computeStableEpsilon(v.x), computeStableEpsilon(v.y)), computeStableEpsilon(v.z));
	}

	auto createRayCameraData(const Camera& camera, const Extent& textureExtent) -> RayCameraData
	{
		const auto origin = vec3f{ camera.origin.x, camera.origin.y, camera.origin.z };
		const auto aspect = textureExtent.width / static_cast<float>(textureExtent.height);

		const auto vz = -normalize(camera.at - origin);
		const auto vx = normalize(cross(camera.up, vz));
		const auto vy = normalize(cross(vz, vx));
		const auto focalDistance = length(camera.at - origin);
		const auto minFocalDistance = max(computeStableEpsilon(origin), computeStableEpsilon(vx));

		const auto screenHeight = 2.f * tanf(camera.FoV / 2.f) * max(minFocalDistance, focalDistance);
		const auto vertical = screenHeight * vy;
		const auto horizontal = screenHeight * aspect * vx;
		const auto lowerLeft = -max(minFocalDistance, focalDistance) * vz - 0.5f * vertical - 0.5f * horizontal;

		return { origin, lowerLeft, horizontal, vertical };
	}

	auto orientedBoxToBox(const owl::box3f& box, const LinearSpace3f& orientation) -> owl::box3f
	{
		const auto extent = owl::vec3f{ box.size().x, box.size().y, box.size().z };

		/*
		 *	  p4----p5
		 *	 / |   / |
		 *	p0----p1 |
		 *	|  p6--|-p7
		 *	| /    |/
		 *	p2----p3
		 */
		const auto p0 = orientation * (0.5f * owl::vec3f(-1.0, -1.0, 1.0) * extent);
		const auto p1 = orientation * (0.5f * owl::vec3f(1.0, -1.0, 1.0) * extent);
		const auto p2 = orientation * (0.5f * owl::vec3f(-1.0, -1.0, -1.0) * extent);
		const auto p3 = orientation * (0.5f * owl::vec3f(1.0, -1.0, -1.0) * extent);
		const auto p4 = orientation * (0.5f * owl::vec3f(-1.0, 1.0, 1.0) * extent);
		const auto p5 = orientation * (0.5f * owl::vec3f(1.0, 1.0, 1.0) * extent);
		const auto p6 = orientation * (0.5f * owl::vec3f(-1.0, 1.0, -1.0) * extent);
		const auto p7 = orientation * (0.5f * owl::vec3f(1.0, 1.0, -1.0) * extent);

		auto newBox = owl::box3f{};
		newBox.extend(p0);
		newBox.extend(p1);
		newBox.extend(p2);
		newBox.extend(p3);
		newBox.extend(p4);
		newBox.extend(p5);
		newBox.extend(p6);
		newBox.extend(p7);

		return newBox;
	}

	std::filesystem::path b3dFilePath{};
} // namespace


auto NanoRenderer::prepareGeometry() -> void
{

	const auto context = nanoContext_.context;

	const auto module = owlModuleCreate(context, NanoRenderer_ptx);

	const auto optixirModule = owlModuleCreateFromIR(context, NanoRenderer_optixir, NanoRenderer_optixir_length);

	[[maybe_unused]] const auto volumeGeometryVars =
		std::array{ OWLVarDecl{ "indexBox", OWL_FLOAT3, OWL_OFFSETOF(NanoVdbVolume, indexBox) },
					OWLVarDecl{ "worldAabb", OWL_FLOAT3, OWL_OFFSETOF(NanoVdbVolume, worldAabb) },
					OWLVarDecl{ "indexBox", OWL_AFFINE3F, OWL_OFFSETOF(NanoVdbVolume, transform) },
					OWLVarDecl{ "grid", OWL_BUFFER_POINTER, OWL_OFFSETOF(NanoVdbVolume, grid) } };

	const auto geometryVars =
		std::array{ OWLVarDecl{ "volume", OWL_USER_TYPE(NanoVdbVolume), OWL_OFFSETOF(GeometryData, volume) }

		};

	const auto geometryType =
		owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(GeometryData), geometryVars.data(), geometryVars.size());


	{


		const auto rayGenerationVars = std::array{
			OWLVarDecl{ "frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenerationFoveatedData, frameBufferSize) },
			OWLVarDecl{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenerationFoveatedData, world) },
			OWLVarDecl{ "foveal", OWL_FLOAT2, OWL_OFFSETOF(RayGenerationFoveatedData, foveal) },
			OWLVarDecl{ "resolutionScaleRatio", OWL_FLOAT, OWL_OFFSETOF(RayGenerationFoveatedData, resolutionScaleRatio) },
			OWLVarDecl{ "kernelParameter", OWL_FLOAT, OWL_OFFSETOF(RayGenerationFoveatedData, kernelParameter)}
		};
		const auto rayGen = owlRayGenCreate(context, optixirModule, "rayGenerationFoveated", sizeof(RayGenerationFoveatedData),
			rayGenerationVars.data(), rayGenerationVars.size());
		nanoContext_.rayGenFoveated = rayGen;
	}
	{
		const auto rayGenerationVars = std::array{
		OWLVarDecl{ "frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenerationData, frameBufferSize) },
		OWLVarDecl{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenerationData, world) },
		};
		const auto rayGen = owlRayGenCreate(context, optixirModule, "rayGeneration", sizeof(RayGenerationData),
			rayGenerationVars.data(), rayGenerationVars.size());
		nanoContext_.rayGen = rayGen;
	}


	auto geometry = owlGeomCreate(context, geometryType);

	const auto geometryGroup = owlUserGeomGroupCreate(context, 1, &geometry);
	nanoContext_.worldGeometryGroup = owlInstanceGroupCreate(context, 1, &geometryGroup, nullptr, nullptr,
															 OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_ALLOW_UPDATE);

	// TODO: need better solution, see also bounds kernel in NanoRenderer.cu
	owlGeomSetRaw(geometry, "volume", &runtimeDataSet_.getSelectedData().volume);

	owlGeomTypeSetBoundsProg(geometryType, module, "volumeBounds");
	owlBuildPrograms(context);

	owlGeomSetPrimCount(geometry, 1);

	owlGroupBuildAccel(geometryGroup);
	owlGroupBuildAccel(nanoContext_.worldGeometryGroup);

	owlRayGenSetGroup(nanoContext_.rayGen, "world", nanoContext_.worldGeometryGroup);
	owlRayGenSetGroup(nanoContext_.rayGenFoveated, "world", nanoContext_.worldGeometryGroup);


	owlGeomTypeSetIntersectProg(geometryType, 0, optixirModule, "nano_intersection");
	owlGeomTypeSetClosestHit(geometryType, 0, optixirModule, "nano_closestHit");


	const auto missProgramVars =
		std::array{ OWLVarDecl{ "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color0) },
					OWLVarDecl{ "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color1) } };

	// ----------- create object  ----------------------------
	nanoContext_.missProgram = owlMissProgCreate(context, optixirModule, "miss", sizeof(MissProgramData),
												 missProgramVars.data(), missProgramVars.size());

	// ----------- set variables  ----------------------------
	owlMissProgSet3f(nanoContext_.missProgram, "color0", owl3f{ .8f, 0.f, 0.f });
	owlMissProgSet3f(nanoContext_.missProgram, "color1", owl3f{ .8f, .8f, .8f });

	{
		const auto launchParamsVarsWithStruct = std::array{
			OWLVarDecl{ "cameraData", OWL_USER_TYPE(RayCameraData), OWL_OFFSETOF(LaunchParams, cameraData) },
			OWLVarDecl{ "surfacePointer", OWL_USER_TYPE(cudaSurfaceObject_t),
						OWL_OFFSETOF(LaunchParams, surfacePointer) },
			OWLVarDecl{ "bg.color0", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, bg.color0) },
			OWLVarDecl{ "bg.color1", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, bg.color1) },
			OWLVarDecl{ "bg.fillBox", OWL_BOOL, OWL_OFFSETOF(LaunchParams, bg.fillBox) },
			OWLVarDecl{ "bg.fillColor", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, bg.fillColor) },
			OWLVarDecl{ "colormaps", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, colorMaps) },
			OWLVarDecl{ "coloringInfo.colorMode", OWL_USER_TYPE(ColoringMode),
						OWL_OFFSETOF(LaunchParams, coloringInfo.coloringMode) },
			OWLVarDecl{ "coloringInfo.singleColor", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, coloringInfo.singleColor) },
			OWLVarDecl{ "coloringInfo.selectedColorMap", OWL_FLOAT,
						OWL_OFFSETOF(LaunchParams, coloringInfo.selectedColorMap) },
			OWLVarDecl{ "transferFunctionTexture", OWL_USER_TYPE(cudaTextureObject_t),
						OWL_OFFSETOF(LaunchParams, transferFunctionTexture) },
			OWLVarDecl{ "sampleRemapping", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, sampleRemapping) },
			OWLVarDecl{ "sampleIntegrationMethod", OWL_USER_TYPE(SampleIntegrationMethod),
						OWL_OFFSETOF(LaunchParams, sampleIntegrationMethod) },
			OWLVarDecl{ "volume", OWL_USER_TYPE(NanoVdbVolume), OWL_OFFSETOF(LaunchParams, volume) }
		};

		nanoContext_.launchParams =
			owlParamsCreate(nanoContext_.context, sizeof(LaunchParams), launchParamsVarsWithStruct.data(),
							launchParamsVarsWithStruct.size());
	}

	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context);
}

bool firstFrame = true;

auto NanoRenderer::onRender() -> void
{
	gpuTimers_.nextFrame();

	auto runtimeVolume = runtimeDataSet_.getSelectedData();
	auto& nanoVdbVolume = runtimeVolume.volume;
	const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");
	trs_ = volumeTransform->worldMatTRS * runtimeVolume.renormalizeScale;
	const auto volumeTranslate = AffineSpace3f::translate(-nanoVdbVolume.indexBox.center());
	const auto groupTransform = trs_ * volumeTranslate;
	owlInstanceGroupSetTransform(nanoContext_.worldGeometryGroup, 0, reinterpret_cast<const float*>(&groupTransform));
	owlGroupRefitAccel(nanoContext_.worldGeometryGroup);
	// if (firstFrame)
	//{


	//	const auto volumeTranslate = AffineSpace3f::translate(-nanoVdbVolume.indexBox.center());
	//	const auto groupTransform = trs_ * volumeTranslate;
	//	owlInstanceGroupSetTransform(nanoContext_.worldGeometryGroup, 0,
	//								 reinterpret_cast<const float*>(&groupTransform));
	//	owlGroupRefitAccel(nanoContext_.worldGeometryGroup);
	//	firstFrame = false;
	//}
	{
		debugDraw().drawBox(trs_.p / 2, trs_.p, nanoVdbVolume.indexBox.size(), owl::vec4f(0.1f, 0.82f, 0.15f, 1.0f),
							trs_.l);

		const auto aabbSize = orientedBoxToBox(nanoVdbVolume.indexBox, volumeTransform->worldMatTRS.l).size();
		debugDraw().drawBox(trs_.p / 2, trs_.p, aabbSize, owl::vec4f(0.9f, 0.4f, 0.2f, 0.4f),
							runtimeVolume.renormalizeScale.l);
	}

	debugInfo_.gizmoHelper->drawGizmo(volumeTransform->worldMatTRS);

	const auto colorMapParams = colorMapFeature_->getParamsData();

	using namespace owl::extensions;

	owlParamsSetRaw(nanoContext_.launchParams, "coloringInfo.colorMode", &colorMapParams.mode);
	owlParamsSet4f(nanoContext_.launchParams, "coloringInfo.singleColor", colorMapParams.uniformColor);

	owlParamsSet1f(nanoContext_.launchParams, "coloringInfo.selectedColorMap", colorMapParams.selectedColorMap);

	owlParamsSetRaw(nanoContext_.launchParams, "sampleIntegrationMethod", &guiData.sampleIntegrationMethode);

	owlParamsSetRaw(nanoContext_.launchParams, "volume", &nanoVdbVolume);

	auto renderTargetFeatureParams = renderTargetFeature_->getParamsData();

	owlParamsSetRaw(nanoContext_.launchParams, "colormaps", &colorMapParams.colorMapTexture);
	owlParamsSet2f(nanoContext_.launchParams, "sampleRemapping",
				   owl2f{ guiData.sampleRemapping[0], guiData.sampleRemapping[1] });

	auto transferFunctionParams = transferFunctionFeature_->getParamsData();

	owlParamsSetRaw(nanoContext_.launchParams, "transferFunctionTexture",
					&transferFunctionParams.transferFunctionTexture);

	const auto backgroundColorParams = backgroundColorFeature_->getParamsData();
	owlParamsSet4f(nanoContext_.launchParams, "bg.color0", backgroundColorParams.colors[0]);
	owlParamsSet4f(nanoContext_.launchParams, "bg.color1", backgroundColorParams.colors[1]);

	owlParamsSet1b(nanoContext_.launchParams, "bg.fillBox", guiData.fillBox);
	owlParamsSet3f(nanoContext_.launchParams, "bg.fillColor",
				   owl3f{ guiData.fillColor[0], guiData.fillColor[1], guiData.fillColor[2] });

	constexpr auto deviceId = 0; //TODO: Research on device id, in multi gpu system it might be tricky
	const auto stream = owlParamsGetCudaStream(nanoContext_.launchParams, deviceId);
	const auto record = gpuTimers_.record("basic owl rt", stream);

	const auto view = renderData_->get<View>("view");
	assert(view->cameras.size() > 0);

	auto rayCameraData = std::array<RayCameraData, 2>{};
	const auto cameraIndices = view->mode == RenderMode::stereo ? std::vector{ 0,1 } : std::vector{ 0 };
	assert(view->mode == RenderMode::stereo ? renderTargetFeatureParams.colorRT.extent.depth > 1: true);

	const auto framebufferSize = owl2i{ static_cast<int32_t>(renderTargetFeatureParams.colorRT.extent.width),
							   static_cast<int32_t>(renderTargetFeatureParams.colorRT.extent.height) };

	for (const auto cameraIndex : cameraIndices)
	{
		assert(cameraIndex < view->cameras.size());
		const auto& camera = view->cameras[cameraIndex];

		if (camera.directionsAvailable)
		{
			rayCameraData[cameraIndex] = { camera.origin, camera.dir00, camera.dirDu, camera.dirDv };
		}
		else
		{
			rayCameraData[cameraIndex] = createRayCameraData(camera,
				Extent
				{
					static_cast<uint32_t>(framebufferSize.x),
					static_cast<uint32_t>(framebufferSize.y),
					1
				}/*renderTargetFeatureParams.colorRT.extent*/);
		}
	}

	const auto foveatedRenderingParams = foveatedFeature_->getControlData();

	const auto foveatedGaze = std::array{
		owl2f{ foveatedRenderingParams.leftEyeGazeScreenSpace.x, foveatedRenderingParams.leftEyeGazeScreenSpace.y },
		owl2f{ foveatedRenderingParams.rightEyeGazeScreenSpace.x, foveatedRenderingParams.rightEyeGazeScreenSpace.y }
	};
	record.start();
	for (const auto cameraIndex : cameraIndices)
	{
		assert(cameraIndex < renderTargetFeatureParams.colorRT.surfaces.size());
		owlParamsSetRaw(nanoContext_.launchParams, "cameraData", &rayCameraData[cameraIndex]);

		if (foveatedRenderingParams.isEnabled)
		{
			owlRayGenSet2f(nanoContext_.rayGenFoveated, "foveal", foveatedGaze[cameraIndex]);
			owlRayGenSet1f(nanoContext_.rayGenFoveated, "resolutionScaleRatio", foveatedFeature_->getResolutionScaleRatio());
			owlRayGenSet1f(nanoContext_.rayGenFoveated, "kernelParameter", foveatedRenderingParams.kernelParameter);
			const auto lpResource = foveatedFeature_->getLpResources()[cameraIndex];

			const auto lrSize = owl2i{ static_cast<int32_t>(lpResource.surface.width),
								   static_cast<int32_t>(lpResource.surface.height) };
			owlRayGenSet2i(nanoContext_.rayGenFoveated, "frameBufferSize", lrSize);
			owlParamsSetRaw(nanoContext_.launchParams, "surfacePointer", &lpResource.surface.surface);
			owlBuildSBT(nanoContext_.context);
			owlAsyncLaunch2D(nanoContext_.rayGenFoveated, lrSize.x, lrSize.y, nanoContext_.launchParams);

			foveatedFeature_->resolve(renderTargetFeatureParams.colorRT.surfaces[cameraIndex], framebufferSize.x, framebufferSize.y, stream, foveatedGaze[cameraIndex].x, foveatedGaze[cameraIndex].y);
		}
		else
		{
			//TODO: pass enable foveated flag to the kernel!
			owlParamsSetRaw(nanoContext_.launchParams, "surfacePointer", &renderTargetFeatureParams.colorRT.surfaces[cameraIndex].surface);
			owlRayGenSet2i(nanoContext_.rayGen, "frameBufferSize", framebufferSize);
			owlBuildSBT(nanoContext_.context);
			owlAsyncLaunch2D(nanoContext_.rayGen, framebufferSize.x, framebufferSize.y, nanoContext_.launchParams);
		}
	}
	record.stop();
}

auto NanoRenderer::onInitialize() -> void
{
	RendererBase::onInitialize();

	const auto context = owlContextCreate(nullptr, 1);

	nanoContext_.context = context;

	//runtimeDataSet_.addNanoVdb(std::filesystem::path{ "D:/datacubes/n4565_cut/nano_level_0_224_257_177.nvdb" });

	prepareGeometry();
}

auto NanoRenderer::onDeinitialize() -> void
{
	owlContextDestroy(nanoContext_.context);
}

auto NanoRenderer::onGui() -> void
{
	const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");

	ImGui::Begin("RT Settings");

	ImGui::SeparatorText("Runtime Data Managment");

	if (ImGui::Button("Load New NanoVDB File"))
	{
		openFileDialog_.open();
	}

	const auto& selectedFiles = openFileDialog_.getSelectedItems();

	if (!selectedFiles.empty())
	{
		for (const auto& selectedFile : selectedFiles)
		{
			runtimeDataSet_.addNanoVdb(selectedFile);
		}
		openFileDialog_.clearSelection();
	}


	for (const auto index : runtimeDataSet_.getValideVolumeIndicies())
	{
		if (ImGui::Button(std::format("Set {}##{}", index, index).c_str()))
		{
			runtimeDataSet_.select(index);
			const auto& statistics = runtimeDataSet_.getStatistics(index);
			guiData.sampleRemapping[0] = statistics.min;
			guiData.sampleRemapping[1] = statistics.max;
		}
		ImGui::SameLine();
	}

	ImGui::Dummy({ 0.0, 0.0 });
	ImGui::Separator();

	if (ImGui::Button("spawn box"))
	{
		static auto box = owl::box3f{};
	}

	ImGui::SeparatorText("Integration Method");
	ImGui::BeginGroup();
	ImGui::RadioButton("Maximum Intensity Projection", reinterpret_cast<int*>(&guiData.sampleIntegrationMethode),
					   static_cast<int>(SampleIntegrationMethod::maximumIntensityProjection));
	ImGui::RadioButton("Average Intensity Projection", reinterpret_cast<int*>(&guiData.sampleIntegrationMethode),
					   static_cast<int>(SampleIntegrationMethod::averageIntensityProjection));
	ImGui::RadioButton("Intensity Integration", reinterpret_cast<int*>(&guiData.sampleIntegrationMethode),
					   static_cast<int>(SampleIntegrationMethod::transferIntegration));
	ImGui::EndGroup();

	ImGui::DragFloatRange2("Sample Remapping", &guiData.sampleRemapping[0], &guiData.sampleRemapping[1], 0.0001, -1.0f,
						   1.0f, "%.4f");

	if (ImGui::Button("Reset Model Transform"))
	{
		volumeTransform->worldMatTRS = AffineSpace3f{};
	}
	ImGui::SeparatorText("Data File (.b3d)");
	ImGui::InputText("##source", const_cast<char*>(b3dFilePath.string().c_str()), b3dFilePath.string().size(),
					 ImGuiInputTextFlags_ReadOnly);
	ImGui::SameLine();
	if (ImGui::Button("Select"))
	{
		// ImGui::OpenPopup("FileSelectDialog");
	}
	ImGui::SameLine();
	if (ImGui::Button("Load"))
	{
		if (std::filesystem::exists(b3dFilePath))
		{
			const auto t = cutterParser::load(b3dFilePath.generic_string());

			const auto nanoFile = t.clusters.front().accelerationStructureRoot.nanoVdbFile;
		}
		else
		{
			ImGui::TextColored(ImVec4{ 0.9f, 0.1f, 0.1f, 1.0f }, "Error: Can't load file!");
		}
	}

	ImGui::Checkbox("Fill Box", &guiData.fillBox);

	if (guiData.fillBox)
	{
		ImGui::ColorEdit3("Fill Color", guiData.fillColor.data());
	}

	ImGui::SeparatorText("Timings");

	const auto timing = gpuTimers_.get("basic owl rt");

	static float values[100] = {};
	static auto valuesOffset = 0;

	values[valuesOffset] = timing;
	valuesOffset = (valuesOffset + 1) % IM_ARRAYSIZE(values);

	{
		auto average = 0.0f;
		for (auto n = 0; n < IM_ARRAYSIZE(values); n++)
		{
			average += values[n];
		}
		average /= static_cast<float>(IM_ARRAYSIZE(values));
		ImGui::SetNextItemWidth(-1);
		ImGui::PlotHistogram("##perfGraph", values, IM_ARRAYSIZE(values), valuesOffset,
							 std::format("avg {:3.2f} ms", average).c_str(), 0.0f, 16.0f, ImVec2(0, 200.0f));
	}

	ImGui::Text("%1.3f", timing);

	

	static auto currentPath = std::filesystem::current_path();
	static auto selectedPath = std::filesystem::path{};

	const auto center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
	openFileDialog_.gui();
	/*if (ImGui::BeginPopupModal("FileSelectDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		constexpr auto roots = std::array{ "A:/", "B:/", "C:/", "D:/", "E:/", "F:/", "G:/", "H:/", "I:/" };

		for (auto i = 0; i < roots.size(); i++)
		{
			const auto root = std::filesystem::path{ roots[i] };
			if (is_directory(root))
			{
				ImGui::SameLine();
				if (ImGui::Button(roots[i]))
				{
					currentPath = root;
				}
			}
		}
		if (ImGui::BeginListBox("##dirs", ImVec2(ImGui::GetFontSize() * 40, ImGui::GetFontSize() * 16)))
		{
			if (ImGui::Selectable("...", false))
			{
				currentPath = currentPath.parent_path();
			}
			auto i = 0;
			for (auto& dir : std::filesystem::directory_iterator{ currentPath })
			{
				i++;
				const auto path = dir.path();
				if (is_directory(path))
				{
					if (ImGui::Selectable(dir.path().string().c_str(), false))
					{
						currentPath = path;
					}
				}
				if (path.has_extension() && path.extension() == ".b3d")
				{
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.1f, 0.9f, 0.1f, 1.0f));
					if (ImGui::Selectable(dir.path().string().c_str(), dir.path() == selectedPath))
					{
						selectedPath = dir.path();
					}
					ImGui::PopStyleColor();
				}
			}
			ImGui::EndListBox();
		}
		if (ImGui::Button("OK", ImVec2(120, 0)))
		{
			if (!selectedPath.empty() != 0)
			{
				b3dFilePath = selectedPath;
			}
			ImGui::CloseCurrentPopup();
		}
		ImGui::SetItemDefaultFocus();
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0)))
		{
			selectedPath.clear();
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
	*/
	ImGui::End();
}
