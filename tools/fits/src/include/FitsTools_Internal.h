#pragma once
#include <cassert>

#include <fitsio.h>

#include "FitsTools.h"

using namespace b3d::common;

inline auto fitsDeleter(fitsfile* file) -> void
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
[[nodiscard]] inline auto openFitsFile(const std::filesystem::path& fitsFilePath, int accessMode = READWRITE)
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
[[nodiscard]] inline auto getFitsProperties(const UniqueFitsFile& fitsFile) -> b3d::tools::fits::FitsProperties;

/// \brief Returns the x, y , z indices of a 3D array when unflattened.
/// \param axisSize Size of the array in each dimension.
/// \param idx one dimensional index
/// \return 3-dimensional index
[[nodiscard]] inline auto unflattenIndex(const Vec3I axisSize, uint64_t idx) -> Vec3I
{
	const auto x = static_cast<int>(idx % axisSize.x);
	const auto y = static_cast<int>((idx / axisSize.x) % axisSize.y);
	const auto z = static_cast<int> (idx / (axisSize.x * axisSize.y));
	return {x, y, z};
}

/// \brief Extracts the bounding box of the data in the FITS file.
/// \param fitsFile FITS file.
/// \return A Box3I representing the bounding box of the data in the FITS file. Zero indexed, lower and upper are included.
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

