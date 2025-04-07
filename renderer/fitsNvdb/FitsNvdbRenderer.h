#pragma once
#include <RendererBase.h>
#include <owl/owl_host.h>

namespace b3d
{
	namespace renderer
	{

		class TransferFunctionFeature;
		class ColorMapFeature;

		namespace fitsNvdb
		{
			struct RayCameraData;
		}

		class RenderTargetFeature;

		class FitsNvdbRenderer final : public RendererBase
		{
			struct RendererContext
			{
				OWLContext owlContext;
				OWLRayGen rayGen;
				OWLMissProg missProgram;
				OWLGeom geometry;
				OWLGroup geometryGroup;
				OWLGroup worldGeometryGroup;
				OWLLaunchParams launchParams;
				std::vector<OWLModule> modules{};
			};

		public:
			FitsNvdbRenderer();

		protected:
			auto onInitialize() -> void override;
			auto onRender() -> void override;
			auto onDeinitialize() -> void override;

		private:
			owl::box3f fitsBox{ { 0, 0, 0 }, { 4, 4, 4 } };
			owl::vec3f volumeTranslateVec{ -fitsBox.center() };
			owl::vec2i currentFramebufferSize{ -1, -1 };
			RenderTargetFeature* renderTargetFeature_;
			RendererContext context_{};

			ColorMapFeature* colorMapFeature_;
			TransferFunctionFeature* transferFunctionFeature_;
			owl::AffineSpace3f trs_{}; // Transformation matrix for the geometry

			bool hasVolume_{ false };
		};
	} // namespace renderer
} // namespace b3d
