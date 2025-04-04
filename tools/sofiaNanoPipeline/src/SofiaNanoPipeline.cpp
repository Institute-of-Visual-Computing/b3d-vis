#include "SofiaNanoPipeline.h"

#include <Box.h>
#include <NanoTools.h>
#include <SofiaProcessRunner.h>

auto b3d::tools::sofia_nano_pipeline::runSearchAndCreateNvdbSync(sofia::SofiaProcessRunner& processRunner,
																 SofiaNanoPipelineInitialParams pipelineParams)
	-> PipelineResult
{
	PipelineResult result;
	// All paths in sofiaParams should be absolute!
	pipelineParams.sofiaParams.setOrReplace("output.writeMask", "true");

	result.sofiaResult = processRunner.runSofiaSync(pipelineParams.sofiaParams);
	if (!result.sofiaResult.wasSuccess())
	{
		result.message = "SoFiA failed.";
		return result;
	}

	const auto paramsOutputPathStr = pipelineParams.sofiaParams.getStringValue("output.directory").value_or("");
	const auto outputDirectoryPath = paramsOutputPathStr.empty() ? pipelineParams.fitsInputFilePath.parent_path() :
																   std::filesystem::path{ paramsOutputPathStr };
	const auto outputFilesNamePrefix = pipelineParams.sofiaParams.getStringValue("output.filename").value_or("");

	// Get Path to mask file. The mask cube will have the suffix _mask.fits
	const auto maskFilePath = outputDirectoryPath / (outputFilesNamePrefix + "_mask.fits");

	if (!std::filesystem::exists(maskFilePath))
	{
		result.message = "Mask file generated by SoFiA not found.";
		return result;
	}

	// Generate nvdb file with
	// - Fits file
	// - mask file
	// - path to nvdb output file
	result.nanoResult = b3d::tools::nano::convertFitsWithMaskToNano(pipelineParams.fitsInputFilePath, maskFilePath,
																	pipelineParams.outputNvdbFilePath);

	if (!result.nanoResult.wasSuccess())
	{
		result.message = "Failed to create NVDB.";
		return result;
	}

	result.finished = true;
	result.message = "Success";
	result.returnCode = 0;

	return result;
}

auto b3d::tools::sofia_nano_pipeline::runSearchAndUpdateNvdbSync(sofia::SofiaProcessRunner& processRunner,
																 SofiaNanoPipelineUpdateParams pipelineParams)
	-> PipelineResult
{
	PipelineResult result;

	// All paths in sofiaParams should be absolute!
	// sofiaParams.setOrReplace("output.writeMask", "true");

	// Produces _mask-raw.fits.
	pipelineParams.sofiaParams.setOrReplace("output.writeRawMask", "true");

	// TODO: This skips Reliability filter.
	// Disables Linker and all following steps.
	pipelineParams.sofiaParams.setOrReplace("linker.enable", "false");

	const auto paramsOutputPathStr = pipelineParams.sofiaParams.getStringValue("output.directory").value_or("");

	const auto outputDirectoryPath = paramsOutputPathStr.empty() ? pipelineParams.fitsInputFilePath.parent_path() :
																   std::filesystem::path{ paramsOutputPathStr };

	const auto outputFilesNamePrefix = pipelineParams.sofiaParams.getStringValue("output.filename").value_or("");

	// Get Path to mask file. The mask cube will have the suffix _mask-raw.fits
	const auto maskFilePath = outputDirectoryPath / (outputFilesNamePrefix + "_mask-raw.fits");

	if (std::filesystem::exists(maskFilePath))
	{
		std::filesystem::remove(maskFilePath);
	}

	result.sofiaResult = processRunner.runSofiaSync(pipelineParams.sofiaParams, pipelineParams.sofiaWorkingDirectory);
	if (result.sofiaResult.returnCode != 8)
	{
		result.message = "SoFiA failed.";
		result.sofiaResult.fileAvailable = false;
		result.sofiaResult.finished = true;
		if (!std::filesystem::exists(maskFilePath))
		{
			result.sofiaResult.message = "Could not find sources. No mask generated.";
		}
		return result;
	}


	result.sofiaResult.resultFile = maskFilePath.generic_string();

	if (!std::filesystem::exists(maskFilePath))
	{
		result.message = "Mask file generated by SoFiA not found.";
		return result;
	}

	// Run nvdb
	result.nanoResult = b3d::tools::nano::createNanoVdbWithExistingAndSubregion(
		pipelineParams.inputNvdbFilePath, pipelineParams.fitsInputFilePath, pipelineParams.maskInputFilePath,
		maskFilePath, pipelineParams.subRegion.lower, pipelineParams.outputNvdbFilePath);

	if (!result.nanoResult.wasSuccess())
	{
		result.message = "Failed to create NVDB.";
		return result;
	}

	result.finished = true;
	result.message = "Success";
	result.returnCode = 0;

	return result;
}
