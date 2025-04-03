#include <cstdlib>
#include <filesystem>

#include <args.hxx>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/ConsoleInitializer.h>
#include <plog/Log.h>
#include <uuid.h>

#include <FileCatalog.h>
#include <FitsTools.h>
#include <NanoTools.h>
#include <Result.h>
#include <SofiaNanoPipeline.h>
#include <TimeStamp.h>

#include "Project.h"
#include "Request.h"

void AddRequestFromFitsAndMaskAsNanoCommandFunction(args::Subparser& parser)
{
	// mask
	// fits
	// project directory (assuming project.json and catalog.json are available)

	// validate mask & fits
	// get project
	// get catalog

	// build request
	// create directories

	// create nvdb

	// write catalog
	// write project
	args::Positional<std::filesystem::path> projectArgument(parser, "PROJECT_DIRECTORY", "Path to projects directory",
															args::Options::Required);


	args::Positional<std::filesystem::path> maskArgument(parser, "MASK_FILE", "Path to mask file (.fits)");

	args::ValueFlagList<uint32_t> maskOffsetArgument(parser, "MASK_OFFSET", "Offset of mask in source file. x, y, z",
													 { "mo", "mask_offset" });
	parser.Parse();


	bool maskAvailable = true;
	if (maskArgument.Get().empty() || !std::filesystem::is_regular_file(maskArgument.Get()))
	{
		LOG_ERROR << "Mask is not available or not a file. Mask is no longer considered.";
		maskAvailable = false;
	}

	auto maskPath = std::filesystem::path("");
	if (maskAvailable)
	{
		maskPath = maskArgument.Get();
		if (!b3d::tools::fits::isFitsFile(maskPath))
		{
			LOG_ERROR << "Mask is not a FITS file. Mask is no longer considered.";
			maskAvailable = false;
		}
	}

	uint32_t maskOffset[] = { 0, 0, 0 };
	if (maskOffsetArgument.Get().size() != 3)
	{
		LOG_ERROR << "Mask offset is not valid. Mask offset is 0, 0, 0";
	}
	else
	{
		maskOffset[0] = maskOffsetArgument.Get()[0];
		maskOffset[1] = maskOffsetArgument.Get()[1];
		maskOffset[2] = maskOffsetArgument.Get()[2];
	}


	if (projectArgument.Get().empty() || !is_directory(projectArgument.Get()))
	{
		LOG_ERROR << "Projects directory is not a valid directory.";
		LOG_INFO << parser;
		return;
	}

	auto projectDirectoryPath = projectArgument.Get();


	LOG_INFO << "Read Project and Catalog.";
	const auto projectsFilePath = projectDirectoryPath / "project.json";
	auto project = b3d::tools::project::Project{};
	try
	{
		std::ifstream f(projectsFilePath);
		const auto data = nlohmann::json::parse(f);
		project = data.get<b3d::tools::project::Project>();
		project.projectPathAbsolute = projectDirectoryPath;
		f.close();
	}
	catch (const std::exception& e)
	{
		LOG_ERROR << "Failed to read project.json.";
		LOG_ERROR << e.what();
		LOG_INFO << parser;
		return;
	}

	b3d::tools::project::catalog::FileCatalog catalog =
		b3d::tools::project::catalog::FileCatalog::createOrLoadCatalogInDirectory(project.projectPathAbsolute);
	std::random_device rd;
	auto seed_data = std::array<int, std::mt19937::state_size>{};
	std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
	std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
	std::mt19937 generator(seq);
	uuids::uuid_random_generator gen{ generator };

	const auto sourcePath = catalog.getFilePathAbsolute(project.fitsOriginUUID);

	if (sourcePath.empty() || !std::filesystem::is_regular_file(sourcePath))
	{
		LOG_ERROR << "Source file " << sourcePath << " is not available.";
		LOG_INFO << parser;
		return;
	}
	LOG_INFO << "Build request";
	auto request = b3d::tools::project::Request{ .uuid = uuids::to_string(gen()),
												 .subRegion = { project.fitsOriginProperties.axisDimensions[0],
																project.fitsOriginProperties.axisDimensions[1],
																project.fitsOriginProperties.axisDimensions[2] },
												 .sofiaParameters = {},
												 .result = {},
												 .createdAt = b3d::common::helper::getSecondsSinceEpochUtc() };

	std::error_code ec;

	if (std::filesystem::create_directories(project.projectPathAbsolute / "requests" / request.uuid, ec) && ec)
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


	if (maskAvailable)
	{
		LOG_INFO << "Copy mask";
		if (std::filesystem::create_directories(project.projectPathAbsolute / "requests" / request.uuid / "sofia",
												ec) &&
			ec)
		{
			LOG_ERROR << "Failed to create directory for mask at "
					  << project.projectPathAbsolute / "requests" / request.uuid / "sofia" << ". Don't copy mask.";
		}
		else
		{
			if (!std::filesystem::copy_file(
					maskPath, project.projectPathAbsolute / "requests" / request.uuid / "sofia" / maskPath.filename(),
					ec))
			{
				LOG_ERROR << "Failed to copy mask file to project directory.";
			}
			else
			{
				request.result.sofiaResult.resultFile = catalog.addFilePathAbsolute(
					project.projectPathAbsolute / "requests" / request.uuid / "sofia" / maskPath.filename());
			}
		}
	}
	else
	{
		LOG_INFO << "Bounds are full volume because no mask available.";
		auto bounds = b3d::tools::fits::extractBounds(sourcePath);
		bounds.upper = bounds.upper + b3d::common::Vec3<int>{ 1, 1, 1 };
		request.subRegion = bounds;
		request.result.sofiaResult.fileAvailable = false;
		request.result.sofiaResult.resultFile = "";
	}

	if (std::filesystem::create_directories(project.projectPathAbsolute / "requests" / request.uuid / "nano", ec) && ec)
	{
		LOG_ERROR << "Failed to create directory for nano result at "
				  << project.projectPathAbsolute / "requests" / request.uuid / "nano" << ".";
		return;
	}

	if (maskAvailable)
	{
		// Check dimensions of mask and source files.
		// if they match -> whole volume
		// if they not match -> subregion pipeline.

		const auto sourceDims = b3d::tools::fits::getFitsProperties(sourcePath).axisDimensions;
		const auto maskDims = b3d::tools::fits::getFitsProperties(maskPath).axisDimensions;
		if (sourceDims.size() != maskDims.size())
		{
			LOG_ERROR << std::format("Dimension count ({}) of source and mask file ({}) do not match.",
									 sourceDims.size(), maskDims.size());
			LOG_INFO << parser;
			return;
		}

		bool dimensionsMatching = true;
		for (int i = 0; i < sourceDims.size(); i++)
		{
			if (sourceDims[i] != maskDims[i])
			{
				LOG_ERROR << "Spatial dimensions of source and mask file do not match.";
				LOG_ERROR << "Assume mask is smaller and offset is valid";
				dimensionsMatching = false;
			}
		}

		if (dimensionsMatching)
		{
			LOG_INFO << "Convert full volume to nvdb with mask.";
			request.result.nanoResult = b3d::tools::nano::convertFitsWithMaskToNano(
				sourcePath, maskPath, project.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb");
		}
		else
		{
			LOG_INFO << "Convert volume to nvdb with provided subregion.";
			// Use existing nvdb file to merge with new masked data.
			const auto nvdbPath = catalog.getFilePathAbsolute(project.requests[0].result.nanoResult.resultFile);
			const auto initialMaskDataPath =
				catalog.getFilePathAbsolute(project.requests[0].result.sofiaResult.resultFile);

			const auto outPath = project.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb";
			request.result.nanoResult = b3d::tools::nano::createNanoVdbWithExistingAndSubregion(
				nvdbPath, sourcePath, initialMaskDataPath, maskPath,
				{ static_cast<int>(maskOffset[0]), static_cast<int>(maskOffset[1]), static_cast<int>(maskOffset[2]) },
				outPath);
		}
	}
	else
	{
		LOG_INFO << "Convert full volume to nvdb without mask.";
		request.result.nanoResult = b3d::tools::nano::convertFitsToNano(
			sourcePath, project.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb");
	}

	if (!request.result.nanoResult.wasSuccess())
	{
		LOG_ERROR << "Failed to create NVDB.";
		return;
	}

	request.result.nanoResult.resultFile =
		catalog.addFilePathAbsolute(project.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb");

	project.requests.push_back(request);

	LOG_INFO << "Save catalog and project files";
	catalog.writeCatalog();

	{
		const auto projectFilePath = project.projectPathAbsolute / "project.json";
		std::ofstream ofs(projectFilePath, std::ofstream::trunc);
		nlohmann::json j = project;
		ofs << std::setw(4) << j << std::endl;
		ofs.close();
	}
	LOG_INFO << "Added Request " << request.uuid;
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

	LOG_INFO << "Copy source file";
	if (!std::filesystem::copy_file(sourcePath, project.projectPathAbsolute / project.fitsOriginFileName, ec))
	{
		LOG_ERROR << "Failed to copy source file to project directory.";
		return;
	}
	sourcePath = project.projectPathAbsolute / project.fitsOriginFileName;

	LOG_INFO << "Copy mask file";
	if (!std::filesystem::copy_file(maskPath, project.projectPathAbsolute / maskPath.filename(), ec))
	{
		LOG_ERROR << "Failed to copy mask file to project directory.";
		return;
	}
	maskPath = project.projectPathAbsolute / maskPath.filename();

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


	if (std::filesystem::create_directories(project.projectPathAbsolute / "requests" / request.uuid, ec) && ec)
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
	request.result.sofiaResult.resultFile = catalog.addFilePathAbsolute(maskPath);

	if (std::filesystem::create_directories(project.projectPathAbsolute / "requests" / request.uuid / "nano", ec) && ec)
	{
		LOG_ERROR << "Failed to create directory for nano result at "
				  << project.projectPathAbsolute / "requests" / request.uuid / "nano" << ".";
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

	request.result.nanoResult.resultFile =
		catalog.addFilePathAbsolute(project.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb");

	project.requests.push_back(request);

	LOG_INFO << "Save catalog and project files";
	catalog.writeCatalog();

	{
		const auto projectFilePath = project.projectPathAbsolute / "project.json";
		std::ofstream ofs(projectFilePath, std::ofstream::trunc);
		nlohmann::json j = project;
		ofs << std::setw(4) << j << std::endl;
		ofs.close();
	}

	LOG_INFO << "Project " << project.projectName << " (" << project.projectUUID << ") created.";
	LOG_INFO << "Project path: " << project.projectPathAbsolute;
}

auto main(const int argc, char** argv) -> int
{
	plog::ColorConsoleAppender<plog::TxtFormatter> colorConsoleAppender;
	plog::init(plog::debug, &colorConsoleAppender);

	args::ArgumentParser parser("Manages Projects for B3D Volume renderer. Projects are hosted on the ProjectServer.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::Group commands(parser, "commands");
	args::Command create(commands, "create", "Create a project.", &CreateCommandFunction);
	args::Command add(commands, "add", "Add nvdb generated by fits and mask to an existing project.",
					  &AddRequestFromFitsAndMaskAsNanoCommandFunction);

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
