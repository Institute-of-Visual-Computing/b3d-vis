#pragma once
#include <boost/process.hpp>

#include "FileCatalog.h"
#include "Project.h"



namespace b3d::tools::sofiasearch
{
	auto appendParameterToSoFiARequest(b3d::tools::projectexplorer::Request& request, const std::string& key, const std::string& val)
		-> void;

	struct ProcessResult
	{
		std::string projectIdentifier;
		std::string requestGUID;
		b3d::tools::projectexplorer::Request request;
	};

	struct RequestProcessor
	{
		auto operator()(projectexplorer::Project& project,
						b3d::tools::projectexplorer::Catalog& rootCatalog,
						std::string requestGUID, b3d::tools::projectexplorer::Request sofiaRequest) -> ProcessResult;
		auto runSearchSync(b3d::tools::projectexplorer::Request& request) -> void;

		auto createNvdb(const  projectexplorer::Project& project,
			const b3d::tools::projectexplorer::Catalog& rootCatalog, b3d::tools::projectexplorer::Request& request)
			-> void;
	};
}
