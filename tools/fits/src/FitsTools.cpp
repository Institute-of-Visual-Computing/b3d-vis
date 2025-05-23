#include "FitsTools.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <format>
#include <iostream>
#include <memory>

#include <cfitsio/fitsio.h>

using namespace b3d::common;

namespace
{
	struct FitsReadParams
	{
		size_t voxelCount;
		std::array<long, 3> min;
		std::array<long, 3> max;
		std::array<long, 3> samplingInterval;
	};

	auto createFitsReadParams(const Box3I& searchBox) -> FitsReadParams
	{
		const auto searchBoxSize = searchBox.size() + Vec3I{ 1, 1, 1 };
		std::array<long, 3> samplingInterval = { 1, 1, 1 };
		std::array<long, 3> min = { searchBox.lower.x + 1, searchBox.lower.y + 1, searchBox.lower.z + 1 };
		std::array<long, 3> max = { searchBoxSize.x, searchBoxSize.y, searchBoxSize.z };

		return FitsReadParams{ static_cast<size_t>(searchBoxSize.x) * searchBoxSize.y * searchBoxSize.z, min, max,
							   samplingInterval };
	}

	auto fitsDeleter(fitsfile* file) -> void
	{
		auto status = int{};
		ffclos(file, &status);
		assert(status == 0);
	};

	using UniqueFitsFile = std::unique_ptr<fitsfile, decltype(&fitsDeleter)>;

#define logError(status)                                                                                               \
	do                                                                                                                 \
	{                                                                                                                  \
		std::array<char, 30> errorMsg;                                                                                 \
		fits_get_errstatus(status, errorMsg.data());                                                                   \
		std::cout << errorMsg.data() << std::endl;                                                                     \
	}                                                                                                                  \
	while (0)

	/// \brief Opens a FITS file.
	/// \param fitsFilePath Path to the FITS file.
	/// \param accessMode Can be READONLY or READWRITE
	/// \return UniqueFitsFile
	[[nodiscard]] auto openFitsFile(const std::filesystem::path& fitsFilePath, int accessMode = READWRITE)
		-> UniqueFitsFile
	{
		fitsfile* fitsFilePtr{ nullptr };
		auto fitsError = int{};
		ffopen(&fitsFilePtr, fitsFilePath.generic_string().c_str(), accessMode, &fitsError);
		assert(fitsError == 0);
		return UniqueFitsFile(fitsFilePtr, &fitsDeleter);
	}

	/// \brief Returns the FITS file properties for up to 3 axis.
	/// \param fitsFile FITS file.
	/// \return Properties of the fits file
	[[nodiscard]] auto getFitsProperties(const UniqueFitsFile& fitsFile) -> b3d::tools::fits::FitsProperties;

	/// \brief Returns the x, y , z indices of a 3D array when unflatten.
	/// \param axisSize Size of the array in each dimension.
	/// \param idx one dimensional index
	/// \return 3-dimensional index
	[[nodiscard]] auto unflattenIndex(const Vec3I axisSize, uint64_t idx) -> Vec3I
	{
		const auto x = static_cast<int>(idx % axisSize.x);
		const auto y = static_cast<int>((idx / axisSize.x) % axisSize.y);
		const auto z = static_cast<int>(idx / (axisSize.x * axisSize.y));
		return { x, y, z };
	}

	/// \brief Extracts the bounding box of the data in the FITS file.
	/// \param fitsFile FITS file.
	/// \return A Box3I representing the bounding box of the data in the FITS file. Zero indexed, lower and upper are
	/// included.
	auto extractBounds(const UniqueFitsFile& fitsFile) -> Box3I;

	/// \brief
	/// \param fitsFile
	/// \param searchBox zero indexed box, lower and upper are included.
	/// \return
	auto extractFloats(const UniqueFitsFile& fitsFile, const Box3I& searchBox) -> b3d::tools::fits::ExtractedFloatData;

	/// \brief
	/// \param fitsFile
	/// \param searchBox zero indexed box, lower and upper are included.
	/// \return
	auto extractIntegers(const UniqueFitsFile& fitsFile, const Box3I& searchBox) -> b3d::tools::fits::ExtractedIntData;


	auto writeFitsFileInternal(const std::filesystem::path& fitsFilePath, const Vec3I boxSize, void* data, int imgType,
							   int dataType) -> void;


	auto extractBounds(const UniqueFitsFile& fitsFile) -> Box3I
	{
		auto fitsError = int{};
		int axisCount;
		int imgType;
		long axis[3];
		fits_get_img_param(fitsFile.get(), 3, &imgType, &axisCount, &axis[0], &fitsError);

		assert(fitsError == 0);
		assert(axisCount == 3);

		const auto srcBox =
			Box3I{ { 0, 0, 0 },
				   { static_cast<int>(axis[0]) - 1, static_cast<int>(axis[1]) - 1, static_cast<int>(axis[2]) - 1 } };
		return srcBox;
	}

	[[nodiscard]] auto getFitsProperties(const UniqueFitsFile& fitsFile) -> b3d::tools::fits::FitsProperties
	{
		b3d::tools::fits::FitsProperties fitsProps;

		auto fitsError = int{ 0 };

		auto imgType = 0;
		fits_get_img_param(fitsFile.get(), 3, &imgType, &fitsProps.axisCount, nullptr, &fitsError);
		assert(fitsProps.axisCount > 0);
		assert(fitsError == 0);

		fitsProps.imgType = static_cast<b3d::tools::fits::FitsImageType>(imgType);
		fitsProps.axisDimensions.resize(fitsProps.axisCount);
		fitsProps.axisTypes.resize(fitsProps.axisCount);

		fits_get_img_param(fitsFile.get(), fitsProps.axisCount, &imgType, &fitsProps.axisCount,
						   fitsProps.axisDimensions.data(), &fitsError);

		auto hduCount = 0;
		fits_get_num_hdus(fitsFile.get(), &hduCount, &fitsError);
		assert(fitsError == 0);

		assert(hduCount > 0);

		fits_movabs_hdu(fitsFile.get(), 1, nullptr, &fitsError);
		assert(fitsError == 0);

		// Read ctype for each axis
		{
			std::vector<char> keyValue;
			// Fits TSTRING is 68 chars + \0
			keyValue.resize(69);
			for (auto i = 0; i < fitsProps.axisCount; i++)
			{
				[[maybe_unused]] auto retVal =
					fits_read_key(fitsFile.get(), TSTRING, std::format("CTYPE{}", i + 1).c_str(), keyValue.data(),
								  nullptr, &fitsError);
				if (fitsError != 0)
				{
					logError(fitsError);
					break;
				}
				fitsProps.axisTypes[i] = std::string{ keyValue.data() };
			}
		}

		return fitsProps;
	}

	auto extractFloats(const UniqueFitsFile& fitsFile, const Box3I& searchBox) -> b3d::tools::fits::ExtractedFloatData
	{
		auto extractedData = b3d::tools::fits::ExtractedFloatData{};
		const auto props = getFitsProperties(fitsFile);

		// Zero indexed, lower and upper are included.
		const auto srcBox =
			Box3I{ { 0, 0, 0 },
				   { props.axisDimensions[0] - 1, props.axisDimensions[1] - 1, props.axisDimensions[2] - 1 } };

		extractedData.box = clip(searchBox, srcBox);
		auto fitsReadParams = createFitsReadParams(extractedData.box);
		extractedData.data.resize(fitsReadParams.voxelCount);

		auto nan = 0l;
		{
			auto error = int{};
			fits_read_subset(fitsFile.get(), TFLOAT, fitsReadParams.min.data(), fitsReadParams.max.data(),
							 fitsReadParams.samplingInterval.data(), &nan, extractedData.data.data(), nullptr, &error);
			if (error != 0)
			{
				std::array<char, 30> txt;
				fits_get_errstatus(error, txt.data());
				std::cout << std::format("CFITSIO error: {}", txt.data());
			}
			assert(error == 0);
		}

		return extractedData;
	}

	auto extractIntegers(const UniqueFitsFile& fitsFile, const Box3I& searchBox) -> b3d::tools::fits::ExtractedIntData
	{
		auto extractedData = b3d::tools::fits::ExtractedIntData{};
		const auto props = getFitsProperties(fitsFile);

		// Zero indexed, lower and upper are included.
		const auto srcBox =
			Box3I{ { 0, 0, 0 },
				   { props.axisDimensions[0] - 1, props.axisDimensions[1] - 1, props.axisDimensions[2] - 1 } };

		extractedData.box = clip(searchBox, srcBox);
		auto fitsReadParams = createFitsReadParams(extractedData.box);
		extractedData.data.resize(fitsReadParams.voxelCount);

		auto nan = 0l;
		{
			auto error = int{};
			fits_read_subset(fitsFile.get(), TINT, fitsReadParams.min.data(), fitsReadParams.max.data(),
							 fitsReadParams.samplingInterval.data(), &nan, extractedData.data.data(), nullptr, &error);
			if (error != 0)
			{
				std::array<char, 30> txt;
				fits_get_errstatus(error, txt.data());
				std::cout << std::format("CFITSIO error: {}", txt.data());
			}
			assert(error == 0);
		}

		return extractedData;
	}

	auto writeFitsFile(const std::filesystem::path& fitsFilePath, const Vec3I boxSize, const std::vector<long>& data)
	{
		assert(boxSize.x * boxSize.y * boxSize.z == data.size());
		writeFitsFileInternal(fitsFilePath, boxSize, (void*)data.data(), LONG_IMG, TLONG);
	}

	auto writeFitsFileInternal(const std::filesystem::path& fitsFilePath, const Vec3I boxSize, void* data, int imgType,
							   int dataType) -> void
	{
		auto fitsError = 0;
		fitsfile* fitsFile{ nullptr };

		auto axis =
			std::array{ static_cast<long>(boxSize.x), static_cast<long>(boxSize.y), static_cast<long>(boxSize.z) };

		fits_create_file(&fitsFile, fitsFilePath.generic_string().c_str(), &fitsError);
		fits_create_img(fitsFile, imgType, 3, axis.data(), &fitsError);
		fits_write_img(fitsFile, dataType, 1, axis[0] * axis[1] * axis[2], data, &fitsError);

		fits_close_file(fitsFile, &fitsError);
		fits_report_error(stderr, fitsError);
	}
} // namespace

auto b3d::tools::fits::isFitsFile(const std::filesystem::path& fitsFilePath) -> bool
{
	fitsfile* fitsFilePtr{ nullptr };
	auto fitsError = int{};
	ffopen(&fitsFilePtr, fitsFilePath.generic_string().c_str(), READONLY, &fitsError);

	if (fitsError == 0)
	{
		auto status = int{};
		ffclos(fitsFilePtr, &status);
		assert(status == 0);
		return true;
	}
	return false;
}

auto b3d::tools::fits::extractBounds(const std::filesystem::path& fitsFilePath) -> Box3I
{
	const auto fitsFile = openFitsFile(fitsFilePath, READONLY);
	return ::extractBounds(fitsFile);
}


auto b3d::tools::fits::extractFloats(const std::filesystem::path& fitsFilePath, const Box3I& searchBox)
	-> ExtractedFloatData
{
	const auto fitsFile = openFitsFile(fitsFilePath, READONLY);
	return ::extractFloats(fitsFile, searchBox);
}

auto b3d::tools::fits::extractIntegers(const std::filesystem::path& fitsFilePath, const Box3I& searchBox)
	-> ExtractedIntData
{
	const auto fitsFile = openFitsFile(fitsFilePath, READONLY);
	return ::extractIntegers(fitsFile, searchBox);
}

auto b3d::tools::fits::writeFitsFile(const std::filesystem::path& fitsFilePath, const Vec3I boxSize,
									 const std::vector<float>& data) -> void
{
	assert(boxSize.x * boxSize.y * boxSize.z == data.size());
	::writeFitsFileInternal(fitsFilePath, boxSize, (void*)data.data(), FLOAT_IMG, TFLOAT);
}

auto b3d::tools::fits::applyMask(const std::vector<float>& data, const std::vector<bool>& mask, const float maskedValue)
	-> std::vector<float>
{
	assert(data.size() == mask.size());
	auto result = std::vector<float>{};
	result.resize(data.size());

	for (auto i = 0; i < data.size(); i++)
	{
		result[i] = mask[i] ? data[i] : maskedValue;
	}
	return result;
}

auto b3d::tools::fits::applyMask(const std::vector<float>& data, const std::vector<int>& mask, const float maskedValue)
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

auto b3d::tools::fits::applyMaskInPlace(std::vector<float>& data, const std::vector<int>& mask, const float maskedValue)
	-> void
{
	assert(data.size() == mask.size());
	for (auto i = size_t{ 0 }; i < data.size(); i++)
	{
		data[i] = mask[i] > 0 ? data[i] : maskedValue;
	}
}

auto b3d::tools::fits::applyMask(const std::filesystem::path& fitsDataFilePath,
								 const std::filesystem::path& fitsMaskFilePath, const float maskedValue)
	-> ExtractedFloatData
{
	auto dataFitsFile = openFitsFile(fitsDataFilePath, READONLY);
	auto maskFitsFile = openFitsFile(fitsMaskFilePath, READONLY);

	const auto dataProps = ::getFitsProperties(dataFitsFile);
	const auto maskProps = ::getFitsProperties(maskFitsFile);
	assert(dataProps.axisDimensions == maskProps.axisDimensions);

	auto floatData = ::extractFloats(dataFitsFile, Box3I::maxBox());
	const auto maskData = ::extractIntegers(maskFitsFile, Box3I::maxBox());

	applyMaskInPlace(floatData.data, maskData.data, maskedValue);
	return floatData;
}

auto b3d::tools::fits::getFitsProperties(const std::filesystem::path& fitsFilePath) -> b3d::tools::fits::FitsProperties
{
	const auto fitsFile = openFitsFile(fitsFilePath, READONLY);
	return ::getFitsProperties(fitsFile);
}

auto b3d::tools::fits::getFitsHeaderInfo(const std::filesystem::path& fitsFilePath) -> b3d::tools::fits::FitsHeaderInfo
{
	const auto fitsFile = openFitsFile(fitsFilePath, READONLY);

	auto fitsHeaderInfo = b3d::tools::fits::FitsHeaderInfo{};

	auto readKey = [&](const char* key) -> std::optional<std::string>
	{
		auto valueString = std::optional<std::string>{};
		char* value;
		auto fitsError = 0;

		[[maybe_unused]] auto retVal = fits_read_key_longstr(fitsFile.get(), key, &value, nullptr, &fitsError);
		if (fitsError != 0)
		{
			return valueString;
		}
		valueString = std::string{ value };
		fits_free_memory(value, &fitsError);
		if (fitsError != 0)
		{
			return valueString;
		}

		return valueString;
	};

	fitsHeaderInfo.author = readKey("AUTHOR");
	fitsHeaderInfo.comment = readKey("COMMENT");
	fitsHeaderInfo.fileCreationDate = readKey("DATE");
	fitsHeaderInfo.object = readKey("OBJECT");
	fitsHeaderInfo.observationDate = readKey("DATE-OBS");
	fitsHeaderInfo.observer = readKey("OBSERVER");
	fitsHeaderInfo.originOrganisation = readKey("ORIGIN");

	return fitsHeaderInfo;
}
