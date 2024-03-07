#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>

#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/IO.h>

#include <NanoCutterParser.h>

#include "Common.h"
#include "FitsHelper.h"

#include "BinaryPartitionClusterProcessor.h"


namespace po = boost::program_options;

auto main(int argc, char** argv) -> int
{

	/*
	 *	program	src		|--source_fits_file file
	 *			[-r		|	--regions				regions_files...	]
	 *			[-lower	|	--threshold_filter_min	value				]		We do not know the stored format upfront
	 *																			value will be cast to the source format
	 *value
	 *			[-upper	|	--threshold_filter_max	value				]		same
	 *			[-c		|	--clamp_to_threshold						]
	 *			[-m		|	--masks					masks_files...		]
	 *			[-dst	|	--storage_directory		directory			]		same as source directory otherwise
	 *			[-l		|	--log_level				none |
	 *												essential |
	 *												all					]		all includes performance counters and
	 *stat. essential is by default
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
	/*
	const auto data = extractData(cutterConfig.src);
	const auto& box = data.box;
	const auto& boxSize = box.size();
	auto nvdb = generateNanoVdb(boxSize, -100.0f, 0.0f, data.data);
	const auto path = (cutterConfig.dst / cutterConfig.src.filename()).string();
	nanovdb::io::writeGrid(path + ".nvdb", nvdb, nanovdb::io::Codec::NONE);
	*/
	return EXIT_SUCCESS;

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

#if 0
	// Scale snippet
	const auto scaleSrc = std::filesystem::path{ "D:/datacubes/n4565_cut/filtered_level_0_224_257_177_id_7.fits" };
	const auto scaleDst =
		std::filesystem::path{ "D:/datacubes/n4565_cut/filtered_level_0_224_257_177_id_7_upscale.fits" };

	auto randomEngine = std::default_random_engine{ std::random_device{}() };
	auto distribution = std::uniform_real_distribution(0.0f, 0.005f);

	upscaleFitsData(scaleSrc.generic_string(), scaleDst.generic_string(), Vec3I{ 8, 8, 8 },
					[&](const float value, [[maybe_unused]] const Vec3I& index, [[maybe_unused]] const Vec3I& boxSize) {
						return value + std::abs(value) > std::numeric_limits<float>::epsilon() ?
							distribution(randomEngine) :
							0.0f;
					});
	return 0;

#endif
	const auto map = extractPerClusterBox(cutterConfig.masks.front(), Box3I::maxBox(), Vec3I{});

	auto trees = std::vector<cutterParser::TreeNode>{};
	constexpr auto bigClusterResolveStrategy = CuttingStrategy::binaryPartition;
	auto processorClusterResults = std::vector<ProcessorResult>{};

	switch (bigClusterResolveStrategy)
	{

	case CuttingStrategy::binaryPartition:
		{
			//TODO: select a good default value, optionally pass vie program arguments
			constexpr auto threshold = 16ull * 16ull * 16ull; // 256ull * 256ull * 256ull;

			auto processor = BinaryPartitionClusterProcessor<Downsampler, threshold>(
				cutterConfig.src, cutterConfig.masks.front(), cutterConfig.dst);

			for (const auto& [clusterId, clusterBox] : map)
			{
				const auto result = processor.process(clusterId, clusterBox);
				processorClusterResults.push_back(result);
			}
		}
		break;
	case CuttingStrategy::fitMemoryReq:
		assert(!"Not Implemented yet!");
		break;
	}
#if 0
	for (const auto& [clusterId, clusterBox] : map)
	{
		const auto parentBox = clusterBox;


		const auto mask = extractBinaryClusterMask(cutterConfig.masks.front(), { clusterId }, clusterBox);
		auto data = extractData(cutterConfig.src, clusterBox);

		constexpr auto maskedValue = -100.0f;
		const auto filteredData = applyMask(data.data, mask, maskedValue);

		const auto size = clusterBox.size();

		const auto fitsFileName = std::format("filtered_level_0_{}_{}_{}_id_{}.fits", size.x, size.y, size.z, id);
		const auto fitsPath = (cutterConfig.dst / fitsFileName).string();

		writeFitsFile(fitsPath.c_str(), size, filteredData);

		const auto fileName = std::format("nano_level_0_{}_{}_{}_id_{}.nvdb", size.x, size.y, size.z, id);
		const auto path = (cutterConfig.dst / fileName).string();

		const auto [min, max] = searchMinMaxBounds(filteredData);

		const auto generatedNanoVdb = generateNanoVdb(size, maskedValue, 0.0f, filteredData);
		std::cout << std::format("NanoVdb buffer size: {}bytes", generatedNanoVdb.size()) << std::endl;
		nanovdb::io::writeGrid(path, generatedNanoVdb,
							   nanovdb::io::Codec::NONE); // TODO: enable nanovdb::io::Codec::BLOSC


		totalMemorySize += generatedNanoVdb.size();

		cutterParser::TreeNode node;
		node.nanoVdbFile = fileName;
		node.nanoVdbBufferSize = generatedNanoVdb.size();
		node.aabb.min = { static_cast<float>(clusterBox.lower.x), static_cast<float>(clusterBox.lower.y),
						  static_cast<float>(clusterBox.lower.z) };
		node.aabb.max = { static_cast<float>(clusterBox.upper.x), static_cast<float>(clusterBox.upper.y),
						  static_cast<float>(clusterBox.upper.z) };
		node.minValue = min;
		node.maxValue = max;
		trees.push_back(node);
		id++;
	}
#endif

	auto totalMemorySize = 0ull;
	auto clusters = std::vector<cutterParser::Cluster>{};
	for(const auto processorResult : processorClusterResults)
	{
		clusters.push_back({processorResult.clusterId, processorResult.totalClusterMemorySize, processorResult.clusterNode});
		totalMemorySize += processorResult.totalClusterMemorySize;
	}

	const auto sourceBounds = extractBounds(cutterConfig.src);
	const auto dataSet = cutterParser::B3DDataSet{
		cutterConfig.src.generic_string(),
		totalMemorySize,
		cutterParser::Box{ { static_cast<float>(sourceBounds.lower.x), static_cast<float>(sourceBounds.lower.y),
							 static_cast<float>(sourceBounds.lower.z) },
						   { static_cast<float>(sourceBounds.upper.x), static_cast<float>(sourceBounds.upper.y),
							 static_cast<float>(sourceBounds.upper.z) } },
		cutterParser::PartitionStrategy::binary,
		clusters
	};

	cutterParser::store(cutterConfig.dst / "project.b3d", dataSet);

	return EXIT_SUCCESS;
}
