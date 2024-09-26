#pragma once

#include "SoFiaSearchView.h"

template <typename T>
auto serialize(const T& t, b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
}

namespace myConverter
{
	template<typename T>
	auto to_string(const T& t) -> std::string
	{
		return std::to_string(t);
	}

	template<>
	auto to_string<bool>(const bool& t) -> std::string
	{
		return t ? "True" : "False";
	}
}
template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::PipelineParams>(const SoFiaSearchView::SofiaParamsTyped::PipelineParams& t,
												 b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::PipelineParams{};

	if (t.pedantic != defaultParams.pedantic)
	{
		sofiaParams.setOrReplace("input.pedantic", t.pedantic ? "True" : "False");
	}

	if (t.verbose != defaultParams.verbose)
	{
		sofiaParams.setOrReplace("input.verbose", t.verbose ? "True" : "False");
	}

	if (t.threads != defaultParams.threads)
	{
		sofiaParams.setOrReplace("input.threads", myConverter::to_string(t.threads));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::InputParams>(const SoFiaSearchView::SofiaParamsTyped::InputParams& t,
											  b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::InputParams{};
	if (t.data != defaultParams.data)
	{
		sofiaParams.setOrReplace("input.data", t.data);
	}

	if (t.region != defaultParams.region)
	{
		sofiaParams.setOrReplace("input.region",
								 std::format("{},{},{},{},{},{}", t.region.lower.x, t.region.upper.x, t.region.lower.y,
											 t.region.upper.y, t.region.lower.z, t.region.upper.z));
	}

	if (t.gain != defaultParams.gain)
	{
		sofiaParams.setOrReplace("input.gain", t.gain);
	}

	if (t.noise != defaultParams.noise)
	{
		sofiaParams.setOrReplace("input.noise", t.noise);
	}

	if (t.weights != defaultParams.weights)
	{
		sofiaParams.setOrReplace("input.weights", t.weights);
	}

	if (t.primaryBeam != defaultParams.primaryBeam)
	{
		sofiaParams.setOrReplace("input.primaryBeam", t.primaryBeam);
	}

	if (t.mask != defaultParams.mask)
	{
		sofiaParams.setOrReplace("input.mask", t.mask);
	}

	if (t.invert != defaultParams.invert)
	{
		sofiaParams.setOrReplace("input.invert", myConverter::to_string(t.invert));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::ContsubParams>(const SoFiaSearchView::SofiaParamsTyped::ContsubParams& t,
												b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::ContsubParams{};
	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("contsub.enable", myConverter::to_string(t.enable));
	}

	if (!t.enable)
	{
		return;
	}

	if (t.order != defaultParams.order)
	{
		sofiaParams.setOrReplace("contsub.order", myConverter::to_string(t.order));
	}

	if (t.threshold != defaultParams.threshold)
	{
		sofiaParams.setOrReplace("contsub.threshold", myConverter::to_string(t.threshold));
	}

	if (t.shift != defaultParams.shift)
	{
		sofiaParams.setOrReplace("contsub.shift", myConverter::to_string(t.shift));
	}

	if (t.padding != defaultParams.padding)
	{
		sofiaParams.setOrReplace("contsub.padding", myConverter::to_string(t.padding));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::FlagParams>(const SoFiaSearchView::SofiaParamsTyped::FlagParams& t,
											 b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::FlagParams{};
	if (t.autoMode != defaultParams.autoMode)
	{
		sofiaParams.setOrReplace("flag.auto", t.autoMode);
	}

	if (t.catalog != defaultParams.catalog)
	{
		sofiaParams.setOrReplace("flag.catalog", t.catalog);
	}

	if (t.cube != defaultParams.cube)
	{
		sofiaParams.setOrReplace("flag.cube", t.cube);
	}

	if (t.log != defaultParams.log)
	{
		sofiaParams.setOrReplace("flag.log", myConverter::to_string(t.log));
	}

	if (t.radius != defaultParams.radius)
	{
		sofiaParams.setOrReplace("flag.radius", myConverter::to_string(t.radius));
	}

	if (!t.region.empty())
	{
		std::string regionString;
		for (auto region : t.region)
		{
			regionString += std::format("{},{},{},{},{},{},", region.lower.x, region.upper.x, region.lower.y,
										region.upper.y, region.lower.z, region.upper.z);
		}

		regionString.resize(regionString.size() - 1);
		sofiaParams.setOrReplace("flag.region", regionString);
	}

	if (t.threshold != defaultParams.threshold)
	{
		sofiaParams.setOrReplace("flag.threshold", myConverter::to_string(t.threshold));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::RippleFilterParams>(const SoFiaSearchView::SofiaParamsTyped::RippleFilterParams& t,
													 b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::RippleFilterParams{};
	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("rippleFilter.enable", myConverter::to_string(t.enable));
	}
	if (!t.enable)
	{
		return;
	}

	if (t.gridXY != defaultParams.gridXY)
	{
		sofiaParams.setOrReplace("rippleFilter.gridXY", myConverter::to_string(t.gridXY));
	}

	if (t.gridZ != defaultParams.gridZ)
	{
		sofiaParams.setOrReplace("rippleFilter.gridZ", myConverter::to_string(t.gridZ));
	}

	if (t.interpolate != defaultParams.interpolate)
	{
		sofiaParams.setOrReplace("rippleFilter.interpolate", myConverter::to_string(t.interpolate));
	}

	if (t.statistic != defaultParams.statistic)
	{
		sofiaParams.setOrReplace("rippleFilter.statistic", t.statistic);
	}

	if (t.windowXY != defaultParams.windowXY)
	{
		sofiaParams.setOrReplace("rippleFilter.windowXY", myConverter::to_string(t.windowXY));
	}
	if (t.windowZ != defaultParams.windowZ)
	{
		sofiaParams.setOrReplace("rippleFilter.windowZ", myConverter::to_string(t.windowZ));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::ScaleNoiseParams>(const SoFiaSearchView::SofiaParamsTyped::ScaleNoiseParams& t,
												   b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::ScaleNoiseParams{};
	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("scaleNoise.enable", myConverter::to_string(t.enable));
	}
	if (!t.enable)
	{
		return;
	}

	if (t.fluxRange != defaultParams.fluxRange)
	{
		sofiaParams.setOrReplace("scaleNoise.fluxRange", t.fluxRange);
	}

	if (t.gridXY != defaultParams.gridXY)
	{
		sofiaParams.setOrReplace("scaleNoise.gridXY", myConverter::to_string(t.gridXY));
	}

	if (t.gridZ != defaultParams.gridZ)
	{
		sofiaParams.setOrReplace("scaleNoise.gridZ", myConverter::to_string(t.gridZ));
	}

	if (t.interpolate != defaultParams.interpolate)
	{
		sofiaParams.setOrReplace("scaleNoise.interpolate", myConverter::to_string(t.interpolate));
	}

	if (t.mode != defaultParams.mode)
	{
		sofiaParams.setOrReplace("scaleNoise.mode", t.mode);
	}

	if (t.scfind != defaultParams.scfind)
	{
		sofiaParams.setOrReplace("scaleNoise.scfind", myConverter::to_string(t.scfind));
	}

	if (t.statistic != defaultParams.statistic)
	{
		sofiaParams.setOrReplace("scaleNoise.statistic", t.statistic);
	}

	if (t.windowXY != defaultParams.windowXY)
	{
		sofiaParams.setOrReplace("scaleNoise.windowXY", myConverter::to_string(t.windowXY));
	}

	if (t.windowZ != defaultParams.windowZ)
	{
		sofiaParams.setOrReplace("scaleNoise.windowZ", myConverter::to_string(t.windowZ));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::ScfindParams>(const SoFiaSearchView::SofiaParamsTyped::ScfindParams& t,
											   b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::ScfindParams{};

	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("scfind.enable", myConverter::to_string(t.enable));
	}

	if (!t.enable)
	{
		return;
	}

	if (t.fluxRange != defaultParams.fluxRange)
	{
		sofiaParams.setOrReplace("scfind.fluxRange", t.fluxRange);
	}

	if (t.kernelsXY != defaultParams.kernelsXY)
	{
		std::string kernelsXY;
		for (auto kernel : t.kernelsXY)
		{
			kernelsXY += myConverter::to_string(kernel) + ",";
		}
		kernelsXY.resize(kernelsXY.size() - 1);
		sofiaParams.setOrReplace("scfind.kernelsXY", kernelsXY);
	}

	if (t.kernelsZ != defaultParams.kernelsZ)
	{
		std::string kernelsZ;
		for (auto kernel : t.kernelsZ)
		{
			kernelsZ += myConverter::to_string(kernel) + ",";
		}
		kernelsZ.resize(kernelsZ.size() - 1);
		sofiaParams.setOrReplace("scfind.kernelsZ", kernelsZ);
	}

	if (t.replacement != defaultParams.replacement)
	{
		sofiaParams.setOrReplace("scfind.replacement", myConverter::to_string(t.replacement));
	}

	if (t.statistic != defaultParams.statistic)
	{
		sofiaParams.setOrReplace("scfind.statistic", t.statistic);
	}

	if (t.threshold != defaultParams.threshold)
	{
		sofiaParams.setOrReplace("scfind.threshold", myConverter::to_string(t.threshold));
	}
}


template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::ThresholdParams>(const SoFiaSearchView::SofiaParamsTyped::ThresholdParams& t,
												  b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::ThresholdParams{};
	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("threshold.enable", myConverter::to_string(t.enable));
	}
	if (!t.enable)
	{
		return;
	}

	if (t.fluxRange != defaultParams.fluxRange)
	{
		sofiaParams.setOrReplace("threshold.fluxRange", t.fluxRange);
	}

	if (t.mode != defaultParams.mode)
	{
		sofiaParams.setOrReplace("threshold.mode", t.mode);
	}

	if (t.statistic != defaultParams.statistic)
	{
		sofiaParams.setOrReplace("threshold.statistic", t.statistic);
	}

	if (t.threshold != defaultParams.threshold)
	{
		sofiaParams.setOrReplace("threshold.threshold", myConverter::to_string(t.threshold));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::LinkerParams>(const SoFiaSearchView::SofiaParamsTyped::LinkerParams& t,
											   b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::LinkerParams{};

	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("linker.enable", myConverter::to_string(t.enable));
	}

	if (!t.enable)
	{
		return;
	}

	if (t.keepNegative != defaultParams.keepNegative)
	{
		sofiaParams.setOrReplace("linker.keepNegative", myConverter::to_string(t.keepNegative));
	}

	if (t.maxFill != defaultParams.maxFill)
	{
		sofiaParams.setOrReplace("linker.maxFill", myConverter::to_string(t.maxFill));
	}

	if (t.maxPixels != defaultParams.maxPixels)
	{
		sofiaParams.setOrReplace("linker.maxPixels", myConverter::to_string(t.maxPixels));
	}

	if (t.maxSizeXY != defaultParams.maxSizeXY)
	{
		sofiaParams.setOrReplace("linker.maxSizeXY", myConverter::to_string(t.maxSizeXY));
	}

	if (t.maxSizeZ != defaultParams.maxSizeZ)
	{
		sofiaParams.setOrReplace("linker.maxSizeZ", myConverter::to_string(t.maxSizeZ));
	}

	if (t.minFill != defaultParams.minFill)
	{
		sofiaParams.setOrReplace("linker.minFill", myConverter::to_string(t.minFill));
	}

	if (t.minPixels != defaultParams.minPixels)
	{
		sofiaParams.setOrReplace("linker.minPixels", myConverter::to_string(t.minPixels));
	}

	if (t.minSizeXY != defaultParams.minSizeXY)
	{
		sofiaParams.setOrReplace("linker.minSizeXY", myConverter::to_string(t.minSizeXY));
	}

	if (t.minSizeZ != defaultParams.minSizeZ)
	{
		sofiaParams.setOrReplace("linker.minSizeZ", myConverter::to_string(t.minSizeZ));
	}

	if (t.positivity != defaultParams.positivity)
	{
		sofiaParams.setOrReplace("linker.positivity", myConverter::to_string(t.positivity));
	}

	if (t.radiusXY != defaultParams.radiusXY)
	{
		sofiaParams.setOrReplace("linker.radiusXY", myConverter::to_string(t.radiusXY));
	}

	if (t.radiusZ != defaultParams.radiusZ)
	{
		sofiaParams.setOrReplace("linker.radiusZ", myConverter::to_string(t.radiusZ));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::ReliabilityParams>(const SoFiaSearchView::SofiaParamsTyped::ReliabilityParams& t,
													b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::ReliabilityParams{};
	if (t.autoKernel != defaultParams.autoKernel)
	{
		sofiaParams.setOrReplace("reliability.autoKernel", myConverter::to_string(t.autoKernel));
	}

	if (t.catalog != defaultParams.catalog)
	{
		sofiaParams.setOrReplace("reliability.catalog", t.catalog);
	}

	if (t.debug != defaultParams.debug)
	{
		sofiaParams.setOrReplace("reliability.debug", myConverter::to_string(t.debug));
	}

	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("reliability.enable", myConverter::to_string(t.enable));
	}

	if (t.iterations != defaultParams.iterations)
	{
		sofiaParams.setOrReplace("reliability.iterations", myConverter::to_string(t.iterations));
	}

	if (t.minPixels != defaultParams.minPixels)
	{
		sofiaParams.setOrReplace("reliability.minPixels", myConverter::to_string(t.minPixels));
	}

	if (t.minSNR != defaultParams.minSNR)
	{
		sofiaParams.setOrReplace("reliability.minSNR", myConverter::to_string(t.minSNR));
	}

	if (t.parameters != defaultParams.parameters)
	{
		std::string parameters;
		for (auto parameter : t.parameters)
		{
			parameters += parameter + ",";
		}
		parameters.resize(parameters.size() - 1);
		sofiaParams.setOrReplace("reliability.parameters", parameters);
	}

	if (t.plot != defaultParams.plot)
	{
		sofiaParams.setOrReplace("reliability.plot", myConverter::to_string(t.plot));
	}

	if (t.scaleKernel != defaultParams.scaleKernel)
	{
		sofiaParams.setOrReplace("reliability.scaleKernel", myConverter::to_string(t.scaleKernel));
	}

	if (t.threshold != defaultParams.threshold)
	{
		sofiaParams.setOrReplace("reliability.threshold", myConverter::to_string(t.threshold));
	}

	if (t.tolerance != defaultParams.tolerance)
	{
		sofiaParams.setOrReplace("reliability.tolerance", myConverter::to_string(t.tolerance));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::DilationParams>(const SoFiaSearchView::SofiaParamsTyped::DilationParams& t,
												 b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::DilationParams{};

	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("dilation.enable", myConverter::to_string(t.enable));
	}

	if (!t.enable)
	{
		return;
	}

	if (t.iterationsXY != defaultParams.iterationsXY)
	{
		sofiaParams.setOrReplace("dilation.iterationsXY", myConverter::to_string(t.iterationsXY));
	}

	if (t.iterationsZ != defaultParams.iterationsZ)
	{
		sofiaParams.setOrReplace("dilation.iterationsZ", myConverter::to_string(t.iterationsZ));
	}

	if (t.threshold != defaultParams.threshold)
	{
		sofiaParams.setOrReplace("dilation.threshold", myConverter::to_string(t.threshold));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::ParameterParams>(const SoFiaSearchView::SofiaParamsTyped::ParameterParams& t,
												  b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::ParameterParams{};

	if (t.enable != defaultParams.enable)
	{
		sofiaParams.setOrReplace("parameter.enable", myConverter::to_string(t.enable));
	}

	if (!t.enable)
	{
		return;
	}

	if (t.offset != defaultParams.offset)
	{
		sofiaParams.setOrReplace("parameter.offset", myConverter::to_string(t.offset));
	}

	if (t.physical != defaultParams.physical)
	{
		sofiaParams.setOrReplace("parameter.physical", myConverter::to_string(t.physical));
	}

	if (t.prefix != defaultParams.prefix)
	{
		sofiaParams.setOrReplace("parameter.prefix", t.prefix);
	}

	if (t.wcs != defaultParams.wcs)
	{
		sofiaParams.setOrReplace("parameter.wcs", myConverter::to_string(t.wcs));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped::OutputParams>(const SoFiaSearchView::SofiaParamsTyped::OutputParams& t,
											   b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	const auto defaultParams = SoFiaSearchView::SofiaParamsTyped::OutputParams{};

	if (t.directory != defaultParams.directory)
	{
		sofiaParams.setOrReplace("output.directory", t.directory);
	}

	if (t.filename != defaultParams.filename)
	{
		sofiaParams.setOrReplace("output.filename", t.filename);
	}

	if (t.marginAperture != defaultParams.marginAperture)
	{
		sofiaParams.setOrReplace("output.marginAperture", myConverter::to_string(t.marginAperture));
	}

	if (t.marginCubelets != defaultParams.marginCubelets)
	{
		sofiaParams.setOrReplace("output.marginCubelets", myConverter::to_string(t.marginCubelets));
	}

	if (t.overwrite != defaultParams.overwrite)
	{
		sofiaParams.setOrReplace("output.overwrite", myConverter::to_string(t.overwrite));
	}

	if (t.thresholdMom12 != defaultParams.thresholdMom12)
	{
		sofiaParams.setOrReplace("output.thresholdMom12", myConverter::to_string(t.thresholdMom12));
	}

	if (t.writeCatASCII != defaultParams.writeCatASCII)
	{
		sofiaParams.setOrReplace("output.writeCatASCII", myConverter::to_string(t.writeCatASCII));
	}

	if (t.writeCatSQL != defaultParams.writeCatSQL)
	{
		sofiaParams.setOrReplace("output.writeCatSQL", myConverter::to_string(t.writeCatSQL));
	}

	if (t.writeCatXML != defaultParams.writeCatXML)
	{
		sofiaParams.setOrReplace("output.writeCatXML", myConverter::to_string(t.writeCatXML));
	}

	if (t.writeCubelets != defaultParams.writeCubelets)
	{
		sofiaParams.setOrReplace("output.writeCubelets", myConverter::to_string(t.writeCubelets));
	}

	if (t.writeFiltered != defaultParams.writeFiltered)
	{
		sofiaParams.setOrReplace("output.writeFiltered", myConverter::to_string(t.writeFiltered));
	}

	if (t.writeKarma != defaultParams.writeKarma)
	{
		sofiaParams.setOrReplace("output.writeKarma", myConverter::to_string(t.writeKarma));
	}

	if (t.writeMask != defaultParams.writeMask)
	{
		sofiaParams.setOrReplace("output.writeMask", myConverter::to_string(t.writeMask));
	}

	if (t.writeMask2d != defaultParams.writeMask2d)
	{
		sofiaParams.setOrReplace("output.writeMask2d", myConverter::to_string(t.writeMask2d));
	}

	if (t.writeMoments != defaultParams.writeMoments)
	{
		sofiaParams.setOrReplace("output.writeMoments", myConverter::to_string(t.writeMoments));
	}

	if (t.writeNoise != defaultParams.writeNoise)
	{
		sofiaParams.setOrReplace("output.writeNoise", myConverter::to_string(t.writeNoise));
	}

	if (t.writePV != defaultParams.writePV)
	{
		sofiaParams.setOrReplace("output.writePV", myConverter::to_string(t.writePV));
	}

	if (t.writeRawMask != defaultParams.writeRawMask)
	{
		sofiaParams.setOrReplace("output.writeRawMask", myConverter::to_string(t.writeRawMask));
	}
}

template <>
auto serialize<SoFiaSearchView::SofiaParamsTyped>(const SoFiaSearchView::SofiaParamsTyped& t, b3d::tools::sofia::SofiaParams& sofiaParams) -> void
{
	serialize(t.pipeline, sofiaParams);
	serialize(t.input, sofiaParams);
	serialize(t.contsub, sofiaParams);
	serialize(t.flag, sofiaParams);
	serialize(t.ripple, sofiaParams);
	serialize(t.scaleNoise, sofiaParams);
	serialize(t.scfind, sofiaParams);
	serialize(t.threshold, sofiaParams);
	serialize(t.linker, sofiaParams);
	serialize(t.reliability, sofiaParams);
	serialize(t.dilation, sofiaParams);
	serialize(t.parameter, sofiaParams);
	serialize(t.output, sofiaParams);
}
