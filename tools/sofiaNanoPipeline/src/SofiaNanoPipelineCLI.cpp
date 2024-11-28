#include <cstdlib>

#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/ConsoleInitializer.h>
#include <plog/Log.h>

#include <uuid.h>

#include "FitsTools.h"
#include "Request.h"

#include <filesystem>

#include <args.hxx>

#include "Project.h"
#include "TimeStamp.h"
#include "Result.h"
#include "NanoTools.h"


void CreateNvdbFromDataAndMaskCommandFunction(args::Subparser& parser)
{
	// Parse pathes to fits file containing data and a second fits file which is the mask
	// Spatial dimensions of the files must match
	// Create a new nvdb file from the data and mask with the help of the NanoTools class

	// Setup help for argument parser
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	
	args::Positional<std::filesystem::path> sourceArgument(parser, "SOURCE_FILE", "Path to source file (.fits)",
														   args::Options::Required);
	args::Positional<std::filesystem::path> maskArgument(parser, "MASK_FILE", "Path to mask file (.fits)",
														 args::Options::Required);

	args::Positional<std::filesystem::path> destinationPathArgument(
		parser, "DESTINATION_PATH", "Filename to the nvdb created.", "");

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

	std::filesystem::path destinationPath = destinationPathArgument.Get();
	if (destinationPath.empty())
	{
		// Use default
		destinationPath = sourcePath.parent_path();
	}

	if (is_directory(destinationPath))
	{
		destinationPath = destinationPath / (sourcePath.stem().string() + ".nvdb");
	}


	const auto sourceDims = b3d::tools::fits::getFitsProperties(sourcePath).axisDimensions;
	const auto maskDims = b3d::tools::fits::getFitsProperties(maskPath).axisDimensions;
	if (sourceDims.size() != maskDims.size())
	{
		LOG_ERROR << std::format("Spatial dimensions ({}) of source and mask file ({}) do not match.",
								 sourceDims.size(),
								 maskDims.size());
		LOG_INFO << parser;
		return;
	}

	for (int i = 0; i < sourceDims.size(); i++)
	{
		if (sourceDims[i] != maskDims[i])
		{
			LOG_ERROR << "Spatial dimensions of source and mask file do not match.";
			LOG_INFO << parser;
			return;
		}
	}
	b3d::tools::nano::convertFitsWithMaskToNano(sourcePath, maskPath, destinationPath);
}

void CreateNvdbFromDataCommandFunction(args::Subparser& parser)
{
	// Setup help for argument parser
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });


	args::Positional<std::filesystem::path> sourceArgument(parser, "SOURCE_FILE", "Path to source file (.fits)",
														   args::Options::Required);

	args::Positional<std::filesystem::path> destinationPathArgument(parser, "DESTINATION_PATH",
																	"Filename to the nvdb created.", "");

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

	std::filesystem::path destinationPath = destinationPathArgument.Get();
	if (destinationPath.empty())
	{
		// Use default
		destinationPath = sourcePath.parent_path();
	}

	if (is_directory(destinationPath))
	{
		destinationPath = destinationPath / (sourcePath.stem().string() + ".nvdb");
	}

	const auto sourceDims = b3d::tools::fits::getFitsProperties(sourcePath).axisDimensions;
	b3d::tools::nano::convertFitsToNano(sourcePath, destinationPath);
}

auto main(const int argc, char** argv) -> int
{
	plog::ColorConsoleAppender<plog::TxtFormatter> colorConsoleAppender;
	plog::init(plog::debug, &colorConsoleAppender);

	args::ArgumentParser parser("Create nvdbs from fits and sofia input");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::Group commands(parser, "commands");

	args::Command createWithMask(commands, "fits-mask-to-nvdb", "Create a nvdb with mask.", &CreateNvdbFromDataAndMaskCommandFunction);
	args::Command createWithoutMask(commands, "fits-to-nvdb", "Create a nvdb without mask.",
									&CreateNvdbFromDataCommandFunction);

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
