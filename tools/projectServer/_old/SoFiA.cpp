#include "SoFiA.h"

#include <iostream>

#include "Catalog.h"

#include <boost/process.hpp>

namespace
{
	const std::array<std::string, 9> sofia_return_code_messages = {
		"The pipeline successfully completed without any error.",
		"An unclassified failure occurred.",
		"A NULL pointer was encountered.",
		"A memory allocation error occurred. This could indicate that the data cube is too large for the amount of memory available on the machine.",
		"An array index was found to be out of range.",
		"An error occurred while trying to read or write a file or check if a directory or file is accessible.",
		"The overflow of an integer value occurred.",
		"The pipeline had to be aborted due to invalid user input. This could, e.g., be due to an invalid parameter setting or the wrong input file being provided.",
		"No specific error occurred, but sources were not detected either."
	};

	const std::array<std::string, 102> sofia_parameter_keys = {
		"pipeline.verbose", "pipeline.pedantic", "pipeline.threads",
		//"input.data",
		"input.region", "input.gain", "input.noise", "input.weights", "input.primaryBeam", "input.mask", "input.invert",
		"flag.region", "flag.catalog", "flag.radius", "flag.auto", "flag.threshold", "flag.log", "contsub.enable",
		"contsub.order", "contsub.threshold", "contsub.shift", "contsub.padding", "scaleNoise.enable",
		"scaleNoise.mode", "scaleNoise.statistic", "scaleNoise.fluxRange", "scaleNoise.windowXY", "scaleNoise.windowZ",
		"scaleNoise.gridXY", "scaleNoise.gridZ", "scaleNoise.interpolate", "scaleNoise.scfind", "rippleFilter.enable",
		"rippleFilter.statistic", "rippleFilter.windowXY", "rippleFilter.windowZ", "rippleFilter.gridXY",
		"rippleFilter.gridZ", "rippleFilter.interpolate", "scfind.enable", "scfind.kernelsXY", "scfind.kernelsZ",
		"scfind.threshold", "scfind.replacement", "scfind.statistic", "scfind.fluxRange", "threshold.enable",
		"threshold.threshold", "threshold.mode", "threshold.statistic", "threshold.fluxRange", "linker.enable",
		"linker.radiusXY", "linker.radiusZ", "linker.minSizeXY", "linker.minSizeZ", "linker.maxSizeXY",
		"linker.maxSizeZ", "linker.minPixels", "linker.maxPixels", "linker.minFill", "linker.maxFill",
		"linker.positivity", "linker.keepNegative", "reliability.enable", "reliability.parameters",
		"reliability.threshold", "reliability.scaleKernel", "reliability.minSNR", "reliability.minPixels",
		"reliability.autoKernel", "reliability.iterations", "reliability.tolerance", "reliability.catalog",
		"reliability.plot", "reliability.debug", "dilation.enable", "dilation.iterationsXY", "dilation.iterationsZ",
		"dilation.threshold", "parameter.enable", "parameter.wcs", "parameter.physical", "parameter.prefix",
		"parameter.offset", "output.directory", "output.filename", "output.writeCatASCII", "output.writeCatXML",
		"output.writeCatSQL", "output.writeNoise", "output.writeFiltered", "output.writeMask", "output.writeMask2d",
		"output.writeRawMask", "output.writeMoments", "output.writeCubelets", "output.writePV", "output.writeKarma",
		"output.marginCubelets", "output.thresholdMom12", "output.overwrite"
	};

	const std::array<std::string, 9> sofia_path_parameter_keys = {
		//"input.data",
		"input.gain",	 "input.mask",			"input.noise",		"input.primaryBeam",
		"input.weights", "flag.catalog", "reliability.catalog", "output.directory",
	};

	auto getSofiaResultMessage(const b3d::tools::projectexplorer::SingleResult& resultMessage) -> std::string_view
	{
		if (0 > resultMessage.returnCode || resultMessage.returnCode >= sofia_return_code_messages.size())
		{
			return sofia_return_code_messages[1];
		}
		return sofia_return_code_messages[resultMessage.returnCode];
	}
} // namespace

auto b3d::tools::sofiasearch::appendParameterToSoFiARequest(b3d::tools::projectexplorer::Request& request,
															const std::string& key, const std::string& val) -> void
{
	if (std::ranges::find(sofia_parameter_keys, key) != sofia_parameter_keys.end())
	{
		// is path like
		if (std::ranges::find(sofia_path_parameter_keys, key) != sofia_path_parameter_keys.end())
		{
			// Can't add path variables
			assert(false);
			auto inputStringForPath = val;
			while (inputStringForPath.starts_with(".") || inputStringForPath.starts_with("/") ||
				   inputStringForPath.starts_with("\\"))
			{
				inputStringForPath.erase(0, 1);
			}

			// TODO: path from catalog
			const auto fullPathString = "";
			//(commonRootPath / boost::process::filesystem::path(inputStringForPath)).string();


			request.sofiaSearchParameters.emplace_back(std::format("{}={}", key.c_str(), fullPathString));
		}
		else
		{
			request.sofiaSearchParameters.emplace_back(std::format("{}={}", key.c_str(), val));
		}
	}
}

auto b3d::tools::sofiasearch::RequestProcessor::runSearchSync(b3d::tools::projectexplorer::Request& search) -> void
{
	search.result.sofia = b3d::tools::projectexplorer::SingleResult{};
	auto allParams{ search.preSofiaSearchParameters};
	allParams.insert(allParams.end(), search.sofiaSearchParameters.begin(), search.sofiaSearchParameters.end());
	auto childProcess =
		boost::process::child(boost::process::exe = search.sofiaExecutablePath.string(), boost::process::args = allParams,
				  boost::process::start_dir = search.workingDirectory.string());
	
	if (childProcess.valid())
	{
		childProcess.wait();
		search.result.sofia.returnCode = childProcess.exit_code();
		search.result.sofia.finished = true;
	}
	else
	{
		search.result.sofia.message = "Process not valid";
		return;
	}
	search.result.sofia.message = getSofiaResultMessage(search.result.sofia);
}

auto b3d::tools::sofiasearch::RequestProcessor::createNvdb(const projectexplorer::Project& project,
														   const b3d::tools::projectexplorer::Catalog& rootCatalog,
														   b3d::tools::projectexplorer::Request& request
														   )
	-> void
{
	const auto originFitsPath = rootCatalog.getFilePathAbsolute(project.fitsOriginUUID);
	const auto maskFitsPath = rootCatalog.getFilePathAbsolute(request.result.sofia.fileResultUUID);
	const auto originFitsMask = project.requests.empty() ?
		maskFitsPath :
		rootCatalog.getFilePathAbsolute(project.requests[0].result.sofia.fileResultUUID);

	const auto originNVDB = project.requests.empty() ?
		maskFitsPath :
		rootCatalog.getFilePathAbsolute(project.requests[0].result.nvdb.fileResultUUID);
	
	const auto destinationNvdbPath = request.workingDirectory / "masked.nvdb";
	auto nvdbResult = NANOCUT_OK;
	if (project.requests.empty())
	{
		nvdbResult = ncConvertFitsToNanoVdbWithMask(originFitsPath.string().c_str(), maskFitsPath.string().c_str(), destinationNvdbPath.string().c_str());
	}
	else
	{
		long offset[3] = { static_cast<long>(request.searchRegion.lower.x),
						   static_cast<long>(request.searchRegion.lower.y),
						   static_cast<long>(request.searchRegion.lower.z)};
		nvdbResult = ncConvertFitsToNanoVdbWithBinaryMask(
			originFitsPath.string().c_str(), originFitsMask.string().c_str(), maskFitsPath.string().c_str(), offset,
			originNVDB.string().c_str(),
			destinationNvdbPath.string().c_str());
	}

	request.result.nvdb.finished = true;
	if (nvdbResult != NANOCUT_OK)
	{
		request.result.nvdb.returnCode = nvdbResult;
		request.result.nvdb.message = nvdbResult == NANOCUT_INVALIDE_ARGUMENT ? "Invalid argument" : "Internal Error";
		return;
	}
	request.result.nvdb.message = "Ok";
	request.result.nvdb.returnCode = 0;
	
	// ncConvertFitsToNanoVdbWithMask()
	// get full fits, get mask

	// create nvdb
}

auto b3d::tools::sofiasearch::RequestProcessor::operator()(projectexplorer::Project &project,
                                                           b3d::tools::projectexplorer::Catalog &rootCatalog, std::string requestUUID,
                                                           b3d::tools::projectexplorer::Request sofiaRequest) -> ProcessResult
{
	const auto startTime = std::chrono::steady_clock::now();
	ProcessResult pr;
	pr.requestGUID = requestUUID;
	pr.projectIdentifier = project.projectUUID;
	pr.request = std::move(sofiaRequest);

	// Create directory for workingdirectory
	std::filesystem::create_directories(pr.request.workingDirectory);

	runSearchSync(pr.request);
	if (!pr.request.result.sofia.wasSuccess() && pr.request.result.sofia.returnCode != 8)
	{
		pr.request.result.finished = true;
		pr.request.result.returnCode = -1;
		pr.request.result.message = "SoFiA failed";
		return pr;
	}
	// Catalog add mask file
	if (project.requests.empty())
	{
		pr.request.result.sofia.fileResultUUID =
			rootCatalog.addFilePathAbsolute(pr.request.workingDirectory / "out_mask.fits");
	}
	else
	{
		pr.request.result.sofia.fileResultUUID =
			rootCatalog.addFilePathAbsolute(pr.request.workingDirectory / "out_mask-raw.fits");
	}
	

	assert(project.fitsOriginProperties.axisCount == 3);

	auto searchRegionSize = pr.request.searchRegion.size();

	createNvdb(project, rootCatalog, pr.request);
	if (!pr.request.result.nvdb.wasSuccess())
	{
		pr.request.result.finished = true;
		pr.request.result.returnCode = -1;
		pr.request.result.message = "Nvdb generation failed";
		return pr;
	}

	pr.request.result.nvdb.fileResultUUID =
		rootCatalog.addFilePathAbsolute(pr.request.workingDirectory / "masked.nvdb");

	pr.request.result.finished = true;
	pr.request.result.returnCode = 0;
	pr.request.result.message = "Request completed";


	const std::chrono::duration<double> requestDuration = std::chrono::steady_clock::now() - startTime;
	auto blubSecs = std::chrono::duration_cast<std::chrono::seconds>(requestDuration);
	pr.request.durationSeconds = blubSecs.count();
	return pr;
}

