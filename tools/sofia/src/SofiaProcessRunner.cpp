#include <boost/process.hpp>

#include "SofiaProcessRunner.h"

// ReSharper disable once CppPassValueParameterByConstReference
auto b3d::tools::sofia::SofiaProcessRunner::runSofiaSync(const SofiaParams params,
														 // ReSharper disable once CppPassValueParameterByConstReference
														 const std::filesystem::path workingDirectory) const
	-> b3d::tools::sofia::SofiaResult
{
	auto childProcess = boost::process::child(
		boost::process::std_out = workingDirectory / "out.log",
		boost::process::std_err = workingDirectory / "err.log",
		boost::process::exe = sofiaExecutablePath_.string(),
		boost::process::args = params.buildCliArguments(),
		boost::process::start_dir = workingDirectory.string());


	b3d::tools::sofia::SofiaResult result;
	if (childProcess.valid())
	{
		childProcess.wait();
		result.returnCode = childProcess.exit_code();
		result.finished = true;
	}
	else
	{
		result.message = "Process not valid";
	}
	result.message = result.getSofiaResultMessage();
	return result;
}


// ReSharper disable once CppPassValueParameterByConstReference
auto b3d::tools::sofia::SofiaProcessRunner::runSofia(const SofiaParams& params,
													 const std::filesystem::path workingDirectory)
	-> std::future<b3d::tools::sofia::SofiaResult>
{
	return std::async(std::launch::async, &SofiaProcessRunner::runSofiaSync, this, params, workingDirectory);
}
