#pragma once

#include <filesystem>
#include <future>

#include <SoFiaResult.h>
#include <SofiaParams.h>

namespace b3d::tools::sofia
{
	class SofiaProcessRunner
	{
	public:
		SofiaProcessRunner(std::filesystem::path sofiaExecutablePath)
			: sofiaExecutablePath_(std::move(sofiaExecutablePath))
		{
		}

		/// \brief Run Sofia with the given parameters.
		/// \return The result as future but resultFile is not set
		auto runSofia(const SofiaParams& params,
					  const std::filesystem::path workingDirectory = std::filesystem::path(".") / "")
			-> std::future<b3d::tools::sofia::SofiaResult>;

		/// \brief Run Sofia with the given parameters.
		/// \return The result but resultFile is not set
		auto runSofiaSync(const SofiaParams params,
						  const std::filesystem::path workingDirectory = std::filesystem::path(".") / "") const
			-> b3d::tools::sofia::SofiaResult;

		// auto runSofia(const SofiaParams& params, std::function<void(const std::string&)> &onOutput) -> void;

	private:
		std::filesystem::path sofiaExecutablePath_;
	};
} // namespace b3d::tools::sofia
