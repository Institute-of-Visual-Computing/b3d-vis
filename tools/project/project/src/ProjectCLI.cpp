#include <cstdlib>

#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/ConsoleInitializer.h>
#include <plog/Log.h>

#include <uuid.h>

#include "FitsTools.h"
#include "Request.h"

#include <filesystem>

#include <args.hxx>

#include "FileCatalog.h"
#include "Project.h"
#include "TimeStamp.h"
#include "Result.h"
#include "NanoTools.h"

namespace
{
}

void CreateCommandFunction(args::Subparser& parser)
{
	/* Create
	 * Required:
	 *   - Source file (.fits)
	 *	 - Mask file (.fits) // XOR Params for SoFiA-2 (If SoFiA-2 is used) [params not implemented yet]
	 * Optional:
	 *   - destination directory (assuming working dir, is not provided)
	 *	 -
	 */


	std::random_device rd;
	auto seed_data = std::array<int, std::mt19937::state_size>{};
	std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
	std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
	std::mt19937 generator(seq);
	uuids::uuid_random_generator gen{ generator };

	std::string projectUUIDString = uuids::to_string(gen());

	std::filesystem::path destinationDirectoryPath = std::filesystem::current_path();

	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::ValueFlag<std::string> projectNameArgument(parser, "PROJECT_NAME", "Name of the project", { 'n', "name" },
													 projectUUIDString);
	args::ValueFlag<std::filesystem::path> destinationDirectoryArgument(
		parser, "DESTINATION_DIRECTORY", "Where to put the project. Using current working directory if not provided.",
		{ 'd', "destination" }, destinationDirectoryPath);
	args::Flag dontCopySourceFilesFlag(parser, "DONT_COPY_SOURCES",
									   "Don't copy source and mask files to the project directory.",
									   { "dc", "dont_copy" });

	args::Positional<std::filesystem::path> sourceArgument(parser, "SOURCE_FILE", "Path to source file (.fits)",
														   args::Options::Required);
	args::Positional<std::filesystem::path> maskArgument(parser, "MASK_FILE", "Path to mask file (.fits)",
														 args::Options::Required);
	parser.Parse();

	if (sourceArgument.Get().empty() || !std::filesystem::is_regular_file(sourceArgument.Get()))
	{
		LOG_ERROR << "Source is not a file.";
		LOG_INFO << parser;
		return;
	}

	auto sourcePath = sourceArgument.Get();

	if (!b3d::tools::fits::isFitsFile(sourcePath))
	{
		LOG_ERROR << "Source is not a FITS file.";
		LOG_INFO << parser;
		return;
	}

	if (maskArgument.Get().empty() || !std::filesystem::is_regular_file(maskArgument.Get()))
	{
		LOG_ERROR << "Mask is not a file.";
		LOG_INFO << parser;
		return;
	}

	auto maskPath = maskArgument.Get();

	if (!b3d::tools::fits::isFitsFile(maskPath))
	{
		LOG_ERROR << "Mask is not a FITS file.";
		LOG_INFO << parser;
		return;
	}

	if (destinationDirectoryArgument.Get().empty() || !is_directory(destinationDirectoryArgument.Get()))
	{
		LOG_ERROR << "Destination directory is not a valid directory.";
		LOG_INFO << parser;
		return;
	}

	destinationDirectoryPath = destinationDirectoryArgument.Get();

	std::string projectName = projectNameArgument.Get();

	// Valid Fits and valid mask file -> generate
	LOG_INFO << "Creating project";
	b3d::tools::project::Project project;
	project.projectName = projectName;
	project.projectUUID = projectUUIDString;
	project.fitsOriginProperties = b3d::tools::fits::getFitsProperties(sourcePath);
	project.fitsOriginFileName = sourcePath.filename().string();
	project.projectPathAbsolute = destinationDirectoryPath / project.projectUUID;

	std::error_code ec;

	if (!std::filesystem::create_directories(project.projectPathAbsolute, ec) && ec)
	{
		LOG_ERROR << "Failed to create project directory at " << project.projectPathAbsolute << ".";
		return;
	}

	if (!dontCopySourceFilesFlag)
	{
		if (!std::filesystem::copy_file(sourcePath, project.projectPathAbsolute / project.fitsOriginFileName, ec))
		{
			LOG_ERROR << "Failed to copy source file to project directory.";
			return;
		}
		sourcePath = project.projectPathAbsolute / project.fitsOriginFileName;
		if (!std::filesystem::copy_file(maskPath, project.projectPathAbsolute / maskPath.filename(), ec))
		{
			LOG_ERROR << "Failed to copy mask file to project directory.";
			return;
		}
		maskPath = project.projectPathAbsolute / maskPath.filename();
	}
	LOG_INFO << "Creating FileCatalog";
	b3d::tools::project::catalog::FileCatalog catalog =
		b3d::tools::project::catalog::FileCatalog::createOrLoadCatalogInDirectory(project.projectPathAbsolute);
	
	project.fitsOriginUUID = catalog.addFilePathAbsolute(project.projectPathAbsolute / project.fitsOriginFileName);
	
	LOG_INFO << "Creating Request";

	auto request = b3d::tools::project::Request{ .uuid = uuids::to_string(gen()),
												 .subRegion = { project.fitsOriginProperties.axisDimensions[0],
																project.fitsOriginProperties.axisDimensions[1],
																project.fitsOriginProperties.axisDimensions[2] },
												 .sofiaParameters = {},
												 .result = {},
												 .createdAt = b3d::common::helper::getSecondsSinceEpochUtc() };


	if (std::filesystem::create_directories(project.projectPathAbsolute / "requests" / request.uuid , ec) &&
		ec)
	{
		LOG_ERROR << "Failed to create request directory at " << project.projectPathAbsolute / "requests" / request.uuid
				  << ".";
		return;
	}

	request.result.returnCode = 0;
	request.result.message = "Success";
	request.result.finished = true;
	request.result.finishedAt = b3d::common::helper::getSecondsSinceEpochUtc();

	request.result.sofiaResult.finished = true;
	request.result.sofiaResult.returnCode = 0;
	request.result.sofiaResult.message = "Success";
	request.result.sofiaResult.finishedAt = b3d::common::helper::getSecondsSinceEpochUtc();
	request.result.sofiaResult.fileAvailable = true;
	request.result.sofiaResult.resultFile = catalog.addFilePathAbsolute(project.projectPathAbsolute / sourcePath.filename());

	if(std::filesystem::create_directories(project.projectPathAbsolute / "requests" / request.uuid / "nano", ec) && ec)
	{
		LOG_ERROR << "Failed to create directory for nano result at " << project.projectPathAbsolute / "requests" / request.uuid / "nano"
				  << ".";
		return;
	}

	LOG_INFO << "Build nvdb";

	request.result.nanoResult = b3d::tools::nano::convertFitsWithMaskToNano(
		sourcePath, maskPath, project.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb");
	if (!request.result.nanoResult.wasSuccess())
	{
		LOG_ERROR << "Failed to create NVDB.";
		return;
	}

	request.result.nanoResult.resultFile = catalog.addFilePathAbsolute(project.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb");

	project.requests.push_back(request);

	catalog.writeCatalog();

	{
		const auto projectFilePath = project.projectPathAbsolute / "project.json";
		std::ofstream ofs(projectFilePath, std::ofstream::trunc);
		nlohmann::json j = project;
		ofs << std::setw(4) << j << std::endl;
		ofs.close();
	}

}

auto main(const int argc, char** argv) -> int
{
	plog::ColorConsoleAppender<plog::TxtFormatter> colorConsoleAppender;
	plog::init(plog::debug, &colorConsoleAppender);

	args::ArgumentParser parser("Manages Projects for B3D Volume renderer. Projects are hosted on the ProjectServer.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::Group commands(parser, "commands");
	args::Command create(commands, "create", "Create a project.", &CreateCommandFunction);

	// args::CompletionFlag completion(parser, { "complete" });
	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (const args::Help&)
	{
		std::cout << parser;
		return EXIT_SUCCESS;
	}
	catch (const args::Error& e)
	{
		LOG_ERROR << e.what();
		LOG_INFO << parser;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
