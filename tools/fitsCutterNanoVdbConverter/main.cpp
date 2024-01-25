#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>

#include <boost/program_options.hpp>
#include <cfitsio/fitsio.h>


#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Primitives.h>

#include <NanoCutterParser.h>

#include "Common.h"
#include "FitsHelper.h"

namespace po = boost::program_options;

auto main(int argc, char** argv) -> int
{

	/*
	 *	program	src		|--source_fits_file file
	 *			[-r		|	--regions				regions_files...	]
	 *			[-lower	|	--threshold_filter_min	value				] //??? We do not know the stored format upfront
	 * //value will be cast to the source format value
	 *			[-upper	|	--threshold_filter_max	value				] //??? same
	 *			[-c		|	--clamp_to_threshold						] //???
	 *			[-m		|	--masks					masks_files...		]
	 *			[-dst	|	--storage_directory		directory			] //same as source directory otherwise
	 *			[-l		|	--log_level				none |
	 *												essential |
	 *												all					] //all includes performance counters and stat.
	 *essential is by default
	 *			=====cutting parameters============
	 *			[-s		|	--strategy				binary_partition |
	 *												fit_memory_req		] //binary_partition by default
	 *			[-f		|	--filter				mean | upper			] //mean by default
	 *			[-mm	|	--max_mem				value_in_bytes		] //upper memory per cut volume. required for
	 *fit_memory_req strategy
	 *			=====refitting=====================
	 *			[-rf	|	--refit_region			aabb				] //only recompute data, that touches the aabb
	 *box inside source fits data
	 *			[-rf_src|	--refit_source_dir		directory			]
	 */


	struct Config
	{
		std::filesystem::path src{};
		std::filesystem::path dst{};
		std::vector<std::filesystem::path> regions{};
		std::vector<std::filesystem::path> masks{};
		LogLevel logLevel{ LogLevel::essential };
		double min{ 0.0 };
		double max{ 0.0 };
		bool clamp{ false };

		CuttingStrategy strategy;
		unsigned int maxMemInBytes;
		Filter filter;
	};

	auto cutterConfig = Config{};


	auto generalConfigurations = po::options_description{ "General Configurations" };
	generalConfigurations.add_options()("source_fits_file,src",
										po::value<std::filesystem::path>(&cutterConfig.src)->required());
	generalConfigurations.add_options()("storage_directory,dst", po::value<std::filesystem::path>(&cutterConfig.dst));
	generalConfigurations.add_options()(
		"regions, r", po::value<std::vector<std::filesystem::path>>(&cutterConfig.regions)->composing());
	generalConfigurations.add_options()(
		"masks, m", po::value<std::vector<std::filesystem::path>>(&cutterConfig.masks)->composing()->multitoken(),
		"mask fits files path");
	generalConfigurations.add_options()("threshold_filter_min,lower", po::value<double>(&cutterConfig.min));
	generalConfigurations.add_options()("threshold_filter_max,upper", po::value<double>(&cutterConfig.max));
	generalConfigurations.add_options()("clamp_to_threshold,c",
										po::value<bool>(&cutterConfig.clamp)->implicit_value(true, "true"));
	generalConfigurations.add_options()(
		"log_level", po::value<LogLevel>(&cutterConfig.logLevel)->default_value(LogLevel::essential, "essential"));
	generalConfigurations.add_options()("help", "produce help message");

	auto cuttingConfigurations = po::options_description{ "Cutting Configurations" };
	cuttingConfigurations.add_options()("strategy",
										po::value<CuttingStrategy>(&cutterConfig.strategy)
											->default_value(CuttingStrategy::binaryPartition, "binary_partition"),
										"possible values are binary_partition and fit_memory_req");
	cuttingConfigurations.add_options()("filter",
										po::value<Filter>(&cutterConfig.filter)->default_value(Filter::mean, "mean"));
	cuttingConfigurations.add_options()(
		"max_mem", po::value<unsigned int>(&cutterConfig.maxMemInBytes)->default_value(0),
		"Max memory in bytes per cut volume. This option should be set for fit_memory_req strategy");

	auto cmdOptions = po::options_description{}.add(generalConfigurations).add(cuttingConfigurations);
	auto vm = po::variables_map{};

	try
	{
		po::store(po::command_line_parser(argc, argv).options(cmdOptions).run(), vm);
		po::notify(vm);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << "\n";
		std::cout << cmdOptions << "\n";
		return EXIT_FAILURE;
	}
	std::cout << cmdOptions << std::endl;


	if (vm.count("help"))
	{
		std::cout << cmdOptions << "\n";
		return EXIT_SUCCESS;
	}


	if (vm.count("source_fits_file"))
	{
		auto isValid = false;
		if (exists(cutterConfig.src) && cutterConfig.src.has_filename())
		{
			isValid = isFitsFile(cutterConfig.src);
		}
		if (!vm.count("storage_directory") && isValid)
		{
			cutterConfig.dst = cutterConfig.src.parent_path() / "cut_output";
		}

		if (!isValid)
		{
			std::cout << "Source Fits file [" << cutterConfig.src.string() << "]"
					  << "is not valid!" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (vm.count("storage_directory"))
	{
		if (!exists(cutterConfig.dst))
		{
			create_directory(cutterConfig.dst);
		}

		const auto isDstValid = is_directory(cutterConfig.dst) &&
			std::filesystem::perms::owner_write ==
				(status(cutterConfig.dst).permissions() & std::filesystem::perms::owner_write);
		if (!isDstValid)
		{
			std::cout << "Storage directory [" << cutterConfig.dst.string() << "]"
					  << "is not valid!" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (vm.count("masks"))
	{
		for (const auto& file : cutterConfig.masks)
		{
			if (!isFitsFile(file))
			{
				std::cout << "Mask file [" << file << "]"
						  << "is not valid!" << std::endl;
				return EXIT_FAILURE;
			}
		}
	}

	const auto fitsFilePath = std::filesystem::path{ "D:/datacubes/testDataSet/n4565.fits" };
	const auto catalogFilePath = std::filesystem::path{ "D:/datacubes/testDataSet/n4565_catalog.fits" };
	const auto maskFilePath = std::filesystem::path{ "D:/datacubes/testDataSet/n4565_mask.fits" };

	const auto map = extractPerClusterBox(cutterConfig.masks.front(), Box3I::maxBox(), Vec3I{});
	// const auto data = extractData(cutterConfig.src, map.at(1));
	// const auto mask = extractBinaryClusterMask(cutterConfig.masks.front(), {19}, map.at(19));//
	// Box3I{{450,410,180},{470,420,200}}); const auto mask2 = extractBinaryClusterMask(cutterConfig.masks.front(),
	// {17}, map.at(17));// Box3I{{450,410,180},{470,420,200}});
	auto trees = std::vector<cutterParser::TreeNode>{};

	for (const auto m : map)
	{
		const auto mask =
			extractBinaryClusterMask(cutterConfig.masks.front(), { m.first },
									 m.second); // Box3I::maxBox());// Box3I{{450,410,180},{470,420,200}});
		auto data = extractData(cutterConfig.src, m.second); // Box3I::maxBox());// Box3I{{450,410,180},{470,420,200}});


		const auto maskedValue = -100.0f;
		const auto filteredData = applyMask(data, mask, maskedValue);

		const auto size = m.second.size();

		static auto ii = 0;

		const auto fitsFileName = std::format("filtered_data_{}_{}_{}_nr{}.fits", size.x, size.y, size.z, ii++);
		const auto fitsPath = (cutterConfig.dst / fitsFileName).string();


		return 0;
		writeFitsFile(fitsPath.c_str(), size, filteredData);

		const auto fileName = std::format("nano_level_0_{}_{}_{}.nvdb", size.x, size.y, size.z);
		const auto path = (cutterConfig.dst / fileName).string();

		generateNanoVdb(path, size, maskedValue, 0.0f, filteredData);


		cutterParser::TreeNode node;
		node.nanoVdbFile = fileName;
		node.aabb.min = { static_cast<float>(m.second.lower.x), static_cast<float>(m.second.lower.y),
						  static_cast<float>(m.second.lower.z) };
		node.aabb.max = { static_cast<float>(m.second.upper.x), static_cast<float>(m.second.upper.y),
						  static_cast<float>(m.second.upper.z) };

		trees.push_back(node);
	}
	cutterParser::store(cutterConfig.dst / "project.b3d", trees);


	return 0;
	fitsfile* fitsFilePtr{ nullptr };
	auto fitsError = int{};
	ffopen(&fitsFilePtr, fitsFilePath.generic_string().c_str(), READONLY, &fitsError);
	assert(fitsError == 0);

	auto fitsFile = UniqueFitsfile(fitsFilePtr, &fitsDeleter);

	int axisCount;
	int imgType;
	long axis[3];
	fits_get_img_param(fitsFile.get(), 3, &imgType, &axisCount, &axis[0], &fitsError);

	assert(fitsError == 0);
	assert(axisCount == 3);
	assert(imgType == FLOAT_IMG);


	const auto box = nanovdb::CoordBBox(nanovdb::Coord(0, 0, 0), nanovdb::Coord(axis[0] - 1, axis[1] - 1, axis[2] - 1));
	auto nan = NAN;


	auto minThreshold = 0.0f;

	long firstPx[3] = { 1, 1, 1 };
	long lastPx[3] = { axis[0], axis[1], axis[2] };
	long inc[3] = { 1, 1, 1 };

	std::vector<float> dataBuffer;
	dataBuffer.resize(axis[0] * axis[1] * axis[2]);


	fits_read_subset(fitsFile.get(), TFLOAT, firstPx, lastPx, inc, &nan, dataBuffer.data(), nullptr, &fitsError);
	if (fitsError != 0)
	{
		std::array<char, 30> txt;
		fits_get_errstatus(fitsError, txt.data());
		std::print(std::cout, "CFITSIO error: {}", txt.data());
	}
	assert(fitsError == 0);

	auto func = [&](const nanovdb::Coord& ijk)
	{
		const auto i = ijk.x();
		const auto j = ijk.y();
		const auto k = ijk.z();
		const auto index = k * axis[0] * axis[1] + j * axis[1] + i;
		const auto v = dataBuffer[index];
		return v > minThreshold ? v : minThreshold;
	};

	// const auto background = 5.0f;

	// const int size = 500;
	//       auto func = [&](const nanovdb::Coord& ijk)
	//{
	//	float v = 40.0f +
	//		50.0f *
	//			(cos(ijk[0] * 0.1f) * sin(ijk[1] * 0.1f) + cos(ijk[1] * 0.1f) * sin(ijk[2] * 0.1f) +
	//			 cos(ijk[2] * 0.1f) * sin(ijk[0] * 0.1f));
	//	v = nanovdb::Max(v, nanovdb::Vec3f(ijk).length() - size); // CSG intersection with a sphere
	//	return v > background ? background : v < -background ? -background : v; // clamp value
	//};


	nanovdb::build::Grid<float> grid(minThreshold, "funny", nanovdb::GridClass::FogVolume);
	grid(func, box);

	auto gridHandle = nanovdb::createNanoGrid(grid);

	std::println(std::cout, "NanoVdb buffer size: {}bytes", gridHandle.size());

	/*auto g = nanovdb::createFogVolumeSphere(10.0f, nanovdb::Vec3d(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0),
	 * "sphere");*/

	nanovdb::io::writeGrid((cutterConfig.dst / "funny.nvdb").string(), gridHandle,
						   nanovdb::io::Codec::NONE); // TODO: enable nanovdb::io::Codec::BLOSC


	cutterParser::TreeNode c1;
	c1.nanoVdbFile = "funny.nvdb";
	cutterParser::TreeNode c2;
	c2.nanoVdbFile = "funny.nvdb";


	cutterParser::TreeNode n;
	n.nanoVdbFile = "funny.nvdb";
	n.children.push_back(c1);
	n.children.push_back(c2);

	cutterParser::store(cutterConfig.dst / "project.b3d", { n });

	int bitpix; // BYTE_IMG (8), SHORT_IMG (16), LONG_IMG (32), LONGLONG_IMG (64), FLOAT_IMG (-32)

	{
		auto status = 0;
		if (fits_get_img_equivtype(fitsFile.get(), &bitpix, &status))
		{
			logError(status);
		}
	}

	{
		auto nan = NAN;
		auto status = 0;
		/*if (fits_read_subset(fitsFile, TFLOAT, firstPx, lastPx, inc, &nan, dataBuffer.data() + nextDataIdx, 0,
		&status))
		{
			logError(status);
		}*/
	}
	// fits_read_subset(fitsFile.get(), type, )

	// fits_get_img_param()
	return EXIT_SUCCESS;
}
