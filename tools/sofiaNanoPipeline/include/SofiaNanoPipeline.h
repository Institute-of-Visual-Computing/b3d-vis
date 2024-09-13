#pragma once
#include <filesystem>

#include "PipelineResult.h"
#include "SofiaParams.h"

namespace b3d::tools::sofia
{
	class SofiaProcessRunner;
}

namespace b3d::tools::sofia_nano_pipeline
{
	struct SofiaNanoPipelineInitialParams
	{
		sofia::SofiaParams sofiaParams;
		std::filesystem::path fitsInputFilePath;
		std::filesystem::path outputNvdbFilePath;
	};

	struct SofiaNanoPipelineUpdateParams
	{
		sofia::SofiaParams sofiaParams;
		// Region the request is defined on. Inside original fits volume.
		// lower inclusive, upper exclusive -> size = upper - lower
		common::Box3I subRegion;
		std::filesystem::path fitsInputFilePath;
		std::filesystem::path maskInputFilePath;
		std::filesystem::path inputNvdbFilePath;
		std::filesystem::path outputNvdbFilePath;
		std::filesystem::path sofiaWorkingDirectoy;
		std::filesystem::path nanoWorkingDirectoy;
	};

	/// \brief Run SoFiA with given params and create a new NVDB based on the output mask
	/// \param processRunner Runner for SoFiA process
	/// \param pipelineParams Params for the pipeline
	/// \return Result of the pipeline
	auto runSearchAndCreateNvdbSync(sofia::SofiaProcessRunner& processRunner,
									SofiaNanoPipelineInitialParams pipelineParams) -> PipelineResult;

	/// \brief Run SoFiA with given params for a subregion. Update an existing nvdb with the potential new sources found in the subregion
	/// \param processRunner Runner for SoFiA process
	/// \param pipelineParams Params for the pipeline
	/// \return Result of the pipeline
	auto runSearchAndUpdateNvdbSync(sofia::SofiaProcessRunner& processRunner,
									SofiaNanoPipelineUpdateParams pipelineParams) -> PipelineResult;

} // namespace b3d::tools::sofia_nano_pipeline
