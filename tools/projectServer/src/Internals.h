#pragma once

#include "Request.h"

namespace b3d::tools::projectServer
{
	struct InternalRequest
	{
		b3d::tools::project::Request userRequest;
		std::string projectUUID;
		sofia::SofiaParams internalParams;
		std::filesystem::path workingDirectoryPath;
		std::filesystem::path sofiaOutputDirectoryPath;

		std::filesystem::path fitsDataInputFilePath;
		std::filesystem::path fitsMaskInputFilePath;
		std::filesystem::path inputNvdbFilePath;
		std::filesystem::path outputNvdbFilePath;
		std::filesystem::path nvdbOutputDirectoryPath;

	};
} // namespace b3d::tools::projectServer

