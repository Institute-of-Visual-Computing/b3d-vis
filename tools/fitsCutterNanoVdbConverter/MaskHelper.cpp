#include "MaskHelper.h"

#include <algorithm>
#include <execution>
#include <iostream>
#include <mdspan>

#include <cfitsio/fitsio.h>

auto extractPerClusterBox(const std::filesystem::path& srcFile, const Box3I& searchBox, const Vec3I& perBatchSearchSize)
	-> std::map<ClusterId, Box3I>
{
	fitsfile* fitsFilePtr{ nullptr };
	auto fitsError = int{};
	ffopen(&fitsFilePtr, srcFile.generic_string().c_str(), READONLY, &fitsError);
	assert(fitsError == 0);

	const auto fitsFile = UniqueFitsfile(fitsFilePtr, &fitsDeleter);

	int axisCount;
	int imgType;
	long axis[3];
	fits_get_img_param(fitsFile.get(), 3, &imgType, &axisCount, &axis[0], &fitsError);

	assert(fitsError == 0);
	assert(axisCount == 3);
	assert(imgType <= LONG_IMG);

	// optimal row size = fits_get_rowsize

	const auto srcBox =
		Box3I{ { 0, 0, 0 }, { static_cast<int>(axis[0]), static_cast<int>(axis[1]), static_cast<int>(axis[2]) } };
	const auto newSearchBox = searchBox != Box3I{} ? clip(searchBox, srcBox) : srcBox;
	const auto searchBoxSize = newSearchBox.size();


	const auto perSearchBatchBoxSize =
		perBatchSearchSize != Vec3I{} ? min(perBatchSearchSize, searchBoxSize) : searchBoxSize;


	const auto batchSize = Vec3I{ searchBoxSize.x / perSearchBatchBoxSize.x, searchBoxSize.y / perSearchBatchBoxSize.y,
								  searchBoxSize.z / perSearchBatchBoxSize.z };


	auto batchBoxes = std::vector<Box3I>{};

	const auto batchItemCount = batchSize.x * batchSize.y * batchSize.z;

	batchBoxes.resize(batchItemCount);

	const auto boxSpan = std::mdspan(batchBoxes.data(), batchSize.x, batchSize.y, batchSize.z);

	for (auto i = 0; i != boxSpan.extent(0); i++)
	{
		for (auto j = 0; j != boxSpan.extent(1); j++)
		{
			for (auto k = 0; k != boxSpan.extent(2); k++)
			{
				boxSpan[std::array{ i, j, k }] = clip(
					Box3I{ perSearchBatchBoxSize * Vec3I(i, j, k), perSearchBatchBoxSize * Vec3I(i + 1, j + 1, k + 1) },
					newSearchBox);
			}
		}
	}

	auto batchResults = std::vector<std::map<ClusterId, Box3I>>{};
	batchResults.resize(batchItemCount);
	auto indices = std::vector<int>{};
	indices.resize(batchItemCount);
	std::iota(indices.begin(), indices.end(), 0);

	auto progressCounter = std::atomic<uint64_t>{ 0 };

	auto loadAndExtract = [&](const int batchItemIndex)
	{
		fitsfile* localFilePtr{ nullptr };
		{
			auto error = int{};
			fits_open_file(&localFilePtr, srcFile.generic_string().c_str(), READONLY, &error);
			assert(error == 0);
		}

		const auto localFitsFile = UniqueFitsfile(localFilePtr, &fitsDeleter);

		const auto box = batchBoxes[batchItemIndex];
		const auto boxSize = box.size();
		const auto voxels = boxSize.x * boxSize.y * boxSize.z;
		std::vector<long> dataBuffer;
		dataBuffer.resize(voxels);

		constexpr auto samplingInterval = std::array{ 1l, 1l, 1l };
		const auto min = std::array<long, 3>{ box.lower.x + 1, box.lower.y + 1, box.lower.z + 1 };
		const auto max = std::array<long, 3>{ box.upper.x, box.upper.y, box.upper.z };
		constexpr auto nan = 0l;

		{
			auto error = int{};
			fits_read_subset(localFitsFile.get(), TINT32BIT, const_cast<long*>(min.data()),
							 const_cast<long*>(max.data()), const_cast<long*>(samplingInterval.data()),
							 const_cast<long*>(&nan), dataBuffer.data(), nullptr, &error);
			if (error != 0)
			{
				std::array<char, 30> txt;
				fits_get_errstatus(error, txt.data());
				std::print(std::cout, "CFITSIO error: {}", txt.data());
			}
			assert(error == 0);
		}


		const auto span = std::mdspan(dataBuffer.data(), boxSize.x, boxSize.y, boxSize.z);

		auto map = std::map<ClusterId, Box3I>{};
		const auto offset = box.lower;
		for (auto i = 0; i != span.extent(0); i++)
		{
			for (auto j = 0; j != span.extent(1); j++)
			{
				for (auto k = 0; k != span.extent(2); k++)
				{
					const auto value = span[std::array{ i, j, k }];
					// skip 0 cluster
					if (value != 0)
					{
						if (map.contains(value))
						{
							map[value].extend(offset + Vec3I{ i, j, k });
						}
						else
						{
							map[value] = Box3I{ offset + Vec3I{ i, j, k }, offset + Vec3I{ i, j, k } };
						}
					}
				}
			}
		}

		batchResults[batchItemIndex] = map;
		progressCounter++;
		progressCounter.notify_one();
	};

	auto progressPrint = [&]()
	{
		auto last = progressCounter.load();
		auto total = batchItemCount;
		auto run = true;
		std::print(std::cout, "processing: {}/{}\r", last, total);

		while (last != total)
		{
			progressCounter.wait(last);
			last = progressCounter.load();

			const auto ratio = last / static_cast<float>(total);

			std::print(std::cout, "processing: {}/{} [{:.2f}%]{}", last, total, ratio * 100.0f,
					   last == total ? "\n" : "\r");
		}
	};

	auto merge = [](const std::map<ClusterId, Box3I>& input1, const std::map<ClusterId, Box3I>& input2)
	{
		std::map<ClusterId, Box3I> merger{};
		for (const auto& [key, value] : input1)
		{
			if (merger.contains(key))
			{
				merger[key].extend(value);
			}
			else
			{
				merger[key] = value;
			}
		}
		for (const auto& [key, value] : input2)
		{
			if (merger.contains(key))
			{
				merger[key].extend(value);
			}
			else
			{
				merger[key] = value;
			}
		}

		return merger;
	};

	// const auto canUseMultithreading = fits_is_reentrant() == 1;
	// const auto executionPolicy = canUseMultithreading ? (std::execution::par_unseq) : (std::execution::seq);

	auto progressPrintThread = std::jthread{ progressPrint };

	std::println(std::cout, "Starting batched data extraction...");
	const auto t1 = std::chrono::high_resolution_clock::now();

	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), loadAndExtract);

	const auto t2 = std::chrono::high_resolution_clock::now();
	const std::chrono::duration<double, std::deci> ms1 = t2 - t1;
	std::println(std::cout, "Data extraction finished in {:.2f} s.", ms1.count());
	std::println(std::cout, "Starting data reducing...");

	auto map = std::reduce(std::execution::par_unseq, batchResults.begin(), batchResults.end(),
						   std::map<ClusterId, Box3I>{}, merge);

	const auto t3 = std::chrono::high_resolution_clock::now();
	const std::chrono::duration<double, std::deci> ms2 = t3 - t2;
	std::println(std::cout, "Data reducing finished in {:.2f} s.", ms2.count());


	return map;
}
