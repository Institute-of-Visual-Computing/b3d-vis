#pragma once

#include "SofiaParams.h"

#include <owl/common/math/AffineSpace.h>

#include "framework/DockableWindowViewBase.h"

class GizmoHelper;

class SoFiaSearchView  final : public DockableWindowViewBase
{
public:

	struct SofiaParamsTyped
	{
		struct PipelineParams
		{
			bool verbose{ false };
			bool pedantic{ true };
			unsigned threads{ 0 };
		};

		struct InputParams
		{
			std::string data;
			owl::box3i region;
			std::string gain;
			std::string noise;
			std::string weights;
			std::string primaryBeam;
			std::string mask;
			bool invert{ false };
		};

		struct FlagParams
		{
			std::string autoMode{ "false" }; // true, false, channels, pixels
			std::vector<owl::box3i> region;
			std::string catalog;
			std::string cube;
			int radius{ 5 };
			float threshold{ 5.0f };
			bool log{ false };
		};

		struct ContsubParams
		{
			bool enable{ false };
			int order{ 0 };
			float threshold{ 2.0f };
			int shift{ 4 };
			int padding{ 3 };
		};

		struct ScaleNoiseParams
		{
			bool enable{ false };
			std::string mode;
			std::string statistic;
			std::string fluxRange;
			int windowXY{ 25 };
			int windowZ{ 15 };
			int gridXY{ 0 };
			int gridZ{ 0 };
			bool interpolate{ false };
			bool scfind{ false };
		};

		struct RippleFilterParams
		{
			bool enable{ false };
			std::string statistic;
			int windowXY{ 31 };
			int windowZ{ 15 };
			int gridXY{ 0 };
			int gridZ{ 0 };
			bool interpolate{ false };
		};

		struct ScfindParams
		{
			bool enable{ true };
			std::vector<int> kernelsXY{ 0, 3, 6 };
			std::vector<int> kernelsZ{ 0, 3, 7, 15 };
			float threshold{ 5.0f };
			float replacement{ 2.0f };
			std::string statistic;
			std::string fluxRange;
		};

		struct ThresholdParams
		{
			bool enable{ false };
			float threshold{ 5.0f };
			std::string mode;
			std::string statistic;
			std::string fluxRange;
		};

		struct LinkerParams
		{
			bool enable{ true };
			int radiusXY{ 1 };
			int radiusZ{ 1 };
			int minSizeXY{ 5 };
			int minSizeZ{ 5 };
			int maxSizeXY{ 0 };
			int maxSizeZ{ 0 };
			int minPixels{ 0 };
			int maxPixels{ 0 };
			float minFill{ 0.0f };
			float maxFill{ 0.0f };
			bool positivity{ false };
			bool keepNegative{ false };
		};

		struct ReliabilityParams
		{
			bool autoKernel{ false };
			bool enable{ false };
			std::vector<std::string> parameters{ "peak", "sum", "mean" };
			float threshold{ 0.9f };
			float scaleKernel{ 0.4f };
			float minSNR{ 3.0f };
			int minPixels{ 0 };
			int iterations{ 30 };
			float tolerance{ 0.05f };
			std::string catalog;
			bool plot{ true };
			bool debug{ false };
		};

		struct DilationParams
		{
			bool enable{ false };
			int iterationsXY{ 10 };
			int iterationsZ{ 5 };
			float threshold{ 0.001f };
		};

		struct ParameterParams
		{
			bool enable{ true };
			bool wcs{ true };
			bool physical{ false };
			std::string prefix;
			bool offset{ false };
		};

		struct OutputParams
		{
			std::string directory;
			std::string filename;
			int marginAperture;
			int marginCubelets;
			bool overwrite{ true };
			float thresholdMom12{ 0.0f };

			bool writeCatASCII{ true };
			bool writeCatSQL{ false };
			bool writeCatXML{ true };
			bool writeCubelets{ false };
			bool writeFiltered{ false };
			bool writeKarma{ false };
			bool writeMask{ false };
			bool writeMask2d{ false };
			bool writeMoments{ false };
			bool writeNoise{ false };
			bool writePV{ false };
			bool writeRawMask{ false };
		};

		PipelineParams pipeline;
		InputParams input;
		ContsubParams contsub;
		FlagParams flag;
		RippleFilterParams ripple;
		ScaleNoiseParams scaleNoise;
		ScfindParams scfind;
		ThresholdParams threshold;
		LinkerParams linker;
		ReliabilityParams reliability;
		DilationParams dilation;
		ParameterParams parameter;
		OutputParams output;

		auto buildSoFiaParams() -> b3d::tools::sofia::SofiaParams;
	};

	

	struct Model
	{
		SofiaParamsTyped params {};
		owl::AffineSpace3f transform{};
		owl::AffineSpace3f worldTransform{};
		owl::box3f selectedLocalRegion{};
		bool showRoiGizmo{ false };
		bool interactionEnabled{ false };
	};

	SoFiaSearchView(ApplicationContext& appContext, Dockspace* dockspace, std::function<void()> startSearchFunction);
	~SoFiaSearchView() override;
	
	auto setModel(Model model) -> void;
	auto getModel() -> Model&;

private:
	auto onDraw() -> void override;
	auto drawFilterFormContent() -> void;

	Model model_;
	std::function<void()> startSearchFunction_;
	std::shared_ptr<GizmoHelper> gizmoHelper_;

	auto resetSelection() -> void;
	auto resetParams() -> void;
};


