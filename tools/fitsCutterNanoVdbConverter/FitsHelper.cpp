#include "FitsHelper.h"

#include <algorithm>
#include <execution>
#include <iostream>

#include <cfitsio/fitsio.h>

// write snippet id taken from Fits samples
auto writeFitsFile(const std::string& file, const Vec3I boxSize, const std::vector<long>& data) -> void
{
	auto fitsFile = fitsfile{};
	int status;
	constexpr auto pixel = 1l;
	constexpr auto axisCount = 3l;
	auto exposure = 0l;
	auto axis = std::array{ static_cast<long>(boxSize.x), static_cast<long>(boxSize.y), static_cast<long>(boxSize.z) };

	status = 0;
	fits_create_file(*fitsFile, file.c_str(), &status);
	fits_create_img(*fitsFile, LONG_IMG, axisCount, axis.data(), &status);
	exposure = 1500.;
	fits_update_key(*fitsFile, TLONG, "EXPOSURE", &exposure, "Total Exposure Time", &status);
	fits_write_img(*fitsFile, TLONG, pixel, data.size(), (void*)data.data(), &status);

	fits_close_file(*fitsFile, &status);
	fits_report_error(stderr, status);
}

auto writeFitsFile(const std::string& file, const Vec3I boxSize, const std::vector<float>& data) -> void
{
	auto fitsFile = fitsfile{};
	int status;
	constexpr auto pixel = 1l;
	constexpr auto axisCount = 3l;
	auto exposure = 0l;
	auto axis = std::array{ static_cast<long>(boxSize.x), static_cast<long>(boxSize.y), static_cast<long>(boxSize.z) };

	status = 0;
	fits_create_file(*fitsFile, file.c_str(), &status);
	fits_create_img(*fitsFile, FLOAT_IMG, axisCount, axis.data(), &status);
	exposure = 1500.;
	fits_update_key(*fitsFile, TLONG, "EXPOSURE", &exposure, "Total Exposure Time", &status);
	fits_write_img(*fitsFile, TLONG, pixel, data.size(), (void*)data.data(), &status);

	fits_close_file(*fitsFile, &status);
	fits_report_error(stderr, status);
}

auto extractPerClusterBox(const std::filesystem::path& srcFile, const Box3I& searchBox, const Vec3I& perBatchSearchSize)
	-> std::map<ClusterId, Box3I>
{
	fitsfile* fitsFilePtr{ nullptr };
	auto fitsError = int{};
	ffopen(&fitsFilePtr, srcFile.generic_string().c_str(), READONLY, &fitsError);
	assert(fitsError == 0);

	int axisCount;
	int imgType;
	long axis[3];
	{
		const auto fitsFile = UniqueFitsfile(fitsFilePtr, &fitsDeleter);
		fits_get_img_param(fitsFile.get(), 3, &imgType, &axisCount, &axis[0], &fitsError);

		assert(fitsError == 0);
		assert(axisCount == 3);
		assert(imgType <= LONG_IMG);

		int bitPerPixel; // BYTE_IMG (8), SHORT_IMG (16), LONG_IMG (32), LONGLONG_IMG (64), FLOAT_IMG (-32)

		{
			auto status = 0;
			if (fits_get_img_equivtype(fitsFile.get(), &bitPerPixel, &status))
			{
				logError(status);
			}
		}
	}

	const auto srcBox =
		Box3I{ { 0, 0, 0 }, { static_cast<int>(axis[0]), static_cast<int>(axis[1]), static_cast<int>(axis[2]) } };
	const auto newSearchBox = searchBox != Box3I{} ? clip(searchBox, srcBox) : srcBox;
	const auto searchBoxSize = newSearchBox.size();


	auto newBatchSize = Vec3I{};
	if (perBatchSearchSize == Vec3I{})
	{
		constexpr auto bytesPerBatch = 6291456; //~64mb
		constexpr auto elementSize = 32; // can be fetched
		const auto bytesPerRow = searchBoxSize.x * elementSize;
		const auto columns = (bytesPerBatch) / bytesPerRow;

		const auto maxConcurrency = static_cast<int>(std::thread::hardware_concurrency());
		const auto uniformBatching = (searchBoxSize.z + maxConcurrency - 1) / maxConcurrency;

		constexpr auto batchesPerThread = 16;

		const auto uniformBatchingPerThread = (uniformBatching + batchesPerThread - 1) / batchesPerThread;

		newBatchSize = Vec3I{ searchBoxSize.x, columns, 1 }; // benchmark 3
		// newBatchSize = Vec3I{ searchBoxSize.x, searchBoxSize.x, 1 };							//benchmark 2
		// newBatchSize = Vec3I{ searchBoxSize.x, searchBoxSize.x, uniformBatchingPerThread };	//benchmark 4
		// newBatchSize = Vec3I{ 64, 64, 64 };												//benchmark 1
	}
	else
	{
		newBatchSize = perBatchSearchSize;
	}

	const auto perSearchBatchBoxSize = min(newBatchSize, searchBoxSize);

	const auto batchSize = Vec3I{ searchBoxSize.x / perSearchBatchBoxSize.x, searchBoxSize.y / perSearchBatchBoxSize.y,
								  searchBoxSize.z / perSearchBatchBoxSize.z };

	auto batchBoxes = std::vector<Box3I>{};

	const auto batchItemCount = batchSize.x * batchSize.y * batchSize.z;

	batchBoxes.resize(batchItemCount);

	for (auto i = 0; i != batchSize.x /*boxSpan.extent(0)*/; i++)
	{
		for (auto j = 0; j != batchSize.y /*boxSpan.extent(1)*/; j++)
		{
			for (auto k = 0; k != batchSize.z /*boxSpan.extent(2)*/; k++)
			{
				const auto index = batchSize.x * batchSize.y * k + batchSize.x * j + i;
				batchBoxes[index] = clip(
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

		auto map = std::map<ClusterId, Box3I>{};
		const auto offset = box.lower;
		for (auto k = 0; k != boxSize.z; k++)
		{
			for (auto j = 0; j != boxSize.y; j++)
			{
				for (auto i = 0; i != boxSize.x; i++)
				{
					const auto index = boxSize.x * boxSize.y * k + boxSize.x * j + i;
					const auto value = dataBuffer[index];
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
		++progressCounter;
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
	const std::chrono::duration<double> ms1 = t2 - t1;
	std::println(std::cout, "Data extraction finished in {:.2f} s.", ms1.count());
	std::println(std::cout, "Starting data reducing...");

	auto map = std::reduce(std::execution::par_unseq, batchResults.begin(), batchResults.end(),
						   std::map<ClusterId, Box3I>{}, merge);

	const auto t3 = std::chrono::high_resolution_clock::now();
	const std::chrono::duration<double> ms2 = t3 - t2;
	std::println(std::cout, "Data reducing finished in {:.2f} s.", ms2.count());
	return map;
}

auto extractBinaryClusterMask(const std::filesystem::path& file, std::vector<ClusterId> clusters,
							  const Box3I& searchBox) -> std::vector<bool>
{
	fitsfile* fitsFilePtr{ nullptr };
	auto fitsError = int{};
	ffopen(&fitsFilePtr, file.generic_string().c_str(), READONLY, &fitsError);
	assert(fitsError == 0);

	const auto fitsFile = UniqueFitsfile(fitsFilePtr, &fitsDeleter);

	int axisCount;
	int imgType;
	long axis[3];
	fits_get_img_param(fitsFile.get(), 3, &imgType, &axisCount, &axis[0], &fitsError);

	assert(fitsError == 0);
	assert(axisCount == 3);
	assert(imgType <= LONG_IMG);

	const auto srcBox =
		Box3I{ { 0, 0, 0 },
			   { static_cast<int>(axis[0]) - 1, static_cast<int>(axis[1]) - 1, static_cast<int>(axis[2]) - 1 } };
	const auto box = searchBox != Box3I{} ? clip(searchBox, srcBox) : srcBox;
	const auto boxSize = box.size();
	const auto voxels =
		static_cast<int64_t>(boxSize.x) * static_cast<int64_t>(boxSize.y) * static_cast<int64_t>(boxSize.z);
	std::vector<int32_t> dataBuffer;
	dataBuffer.resize(voxels);

	constexpr auto samplingInterval = std::array{ 1l, 1l, 1l };
	const auto min = std::array<long, 3>{ box.lower.x + 1, box.lower.y + 1, box.lower.z + 1 };
	const auto max = std::array<long, 3>{ box.upper.x, box.upper.y, box.upper.z };
	auto nan = 0l;
	{
		auto error = int{};
		fits_read_subset(fitsFile.get(), TINT32BIT, const_cast<long*>(min.data()), const_cast<long*>(max.data()),
						 const_cast<long*>(samplingInterval.data()), &nan, dataBuffer.data(), nullptr, &error);
		if (error != 0)
		{
			std::array<char, 30> txt;
			fits_get_errstatus(error, txt.data());
			std::print(std::cout, "CFITSIO error: {}", txt.data());
		}
		assert(error == 0);
	}

	std::vector<bool> maskBuffer;
	maskBuffer.resize(voxels);


	auto filter = [&clusters](const long& value) -> bool
	{
		for (const auto clusterId : clusters)
		{
			if (value == clusterId)
			{
				return true;
			}
		}
		return false;
	};

	std::transform(std::execution::seq, dataBuffer.begin(), dataBuffer.end(), maskBuffer.begin(), filter);

	return maskBuffer;
}

auto extractClusterMask(const std::filesystem::path& file, ClusterId cluster, const Box3I& searchBox)
	-> std::vector<uint32_t>
{
	return std::vector<uint32_t>();
}


auto extractData(const std::filesystem::path& file, const Box3I& searchBox) -> ExtractedData
{
	fitsfile* fitsFilePtr{ nullptr };
	auto fitsError = int{};
	ffopen(&fitsFilePtr, file.generic_string().c_str(), READONLY, &fitsError);
	assert(fitsError == 0);

	const auto fitsFile = UniqueFitsfile(fitsFilePtr, &fitsDeleter);

	auto axisCount = 0;
	auto imgType = 0;
	fits_get_img_param(fitsFile.get(), 3, &imgType, &axisCount, nullptr, &fitsError);
	assert(axisCount > 0);
	auto axis = std::vector<long>{};
	axis.resize(axisCount);
	fits_get_img_param(fitsFile.get(), 3, &imgType, &axisCount, axis.data(), &fitsError);

	assert(fitsError == 0);
	assert(axisCount >= 3);
	assert(imgType <= FLOAT_IMG);

	const auto srcBox =
		Box3I{ { 0, 0, 0 },
			   { static_cast<int>(axis[0]) - 1, static_cast<int>(axis[1]) - 1, static_cast<int>(axis[2]) - 1 } };
	const auto newSearchBox = searchBox != Box3I{} ? clip(searchBox, srcBox) : srcBox;
	const auto searchBoxSize = newSearchBox.size();

	const auto voxels = searchBoxSize.x * searchBoxSize.y * searchBoxSize.z;
	std::vector<float> dataBuffer;
	dataBuffer.resize(voxels);

	auto samplingInterval = std::vector<long>{};
	auto min = std::vector<long>{};
	auto max = std::vector<long>{};
	samplingInterval.resize(axisCount);
	min.resize(axisCount);
	max.resize(axisCount);
	std::ranges::fill(samplingInterval, 1l);
	std::ranges::fill(min, 1l);
	std::ranges::fill(max, 1l);

	min[0] = newSearchBox.lower.x + 1;
	min[1] = newSearchBox.lower.y + 1;
	min[2] = newSearchBox.lower.z + 1;

	max[0] = newSearchBox.upper.x;
	max[1] = newSearchBox.upper.y;
	max[2] = newSearchBox.upper.z;

	{
		auto error = int{};
		fits_read_subset(fitsFile.get(), TFLOAT, min.data(), max.data(), samplingInterval.data(), &nan,
						 dataBuffer.data(), nullptr, &error);
		if (error != 0)
		{
			std::array<char, 30> txt;
			fits_get_errstatus(error, txt.data());
			std::print(std::cout, "CFITSIO error: {}", txt.data());
		}
		assert(error == 0);
	}

	return ExtractedData{ newSearchBox, dataBuffer };
}
auto applyMask(const std::vector<float>& data, const std::vector<bool>& mask, const float maskedValue)
	-> std::vector<float>
{
	assert(data.size() == mask.size());
	auto result = std::vector<float>{};
	result.resize(data.size());

	for (auto i = 0; i < data.size(); i++)
	{
		result[i] = mask[i] > 0 ? data[i] : maskedValue;
	}
	return result;
}

auto searchMinMaxBounds(const std::vector<float>& data) -> MinMaxBounds
{
	MinMaxBounds bounds;
	bounds.min = std::numeric_limits<float>::max();
	bounds.max = std::numeric_limits<float>::min();
	for (auto i = 0; i < data.size(); i++)
	{
		bounds.max = std::max({ bounds.max, data[i] });
		bounds.min = std::min({ bounds.min, data[i] });
	}
	return bounds;
}

auto upscaleFitsData(const std::string& srcFile, const std::string& dstFile, const Vec3I& axisScaleFactor,
					 const std::function<float(const float, const Vec3I&, const Vec3I&)>& filter) -> void
{
	const auto data = extractData(srcFile);

	auto upscaleData = std::vector<float>{};
	const auto newSize = static_cast<size_t>(data.data.size()) * static_cast<size_t>(axisScaleFactor.x) *
		static_cast<size_t>(axisScaleFactor.y) * static_cast<size_t>(axisScaleFactor.z);
	upscaleData.resize(newSize);
	const auto& box = data.box;
	const auto& boxSize = box.size();

	const auto dstBoxSize =
		Vec3I{ boxSize.x * axisScaleFactor.x, boxSize.y * axisScaleFactor.y, boxSize.z * axisScaleFactor.z };


	for (auto i = 0ull; i < dstBoxSize.z; i++)
	{
		for (auto j = 0ull; j < dstBoxSize.y; j++)
		{
			for (auto k = 0ull; k < dstBoxSize.x; k++)
			{
				const auto index1D = i * dstBoxSize.y * dstBoxSize.x + j * dstBoxSize.x + k;

				const auto indexSrc1D = i / axisScaleFactor.z * boxSize.y * boxSize.x +
					j / axisScaleFactor.y * boxSize.x + k / axisScaleFactor.x;
				upscaleData[index1D] =
					filter(data.data[indexSrc1D],
						   Vec3I{ static_cast<int>(i), static_cast<int>(j), static_cast<int>(k) }, dstBoxSize);
			}
		}
	}

	writeFitsFile(dstFile, dstBoxSize, upscaleData);
	generateNanoVdb(dstFile + ".nvdb", dstBoxSize, -100.0f, 0.0, upscaleData);
}
