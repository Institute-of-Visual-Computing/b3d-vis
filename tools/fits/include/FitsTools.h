#pragma once

#include <filesystem>

#include <Box.h>
#include <FitsCommon.h>

namespace b3d::tools::fits
{
	/// \brief Describes extracted data from a FITS file. box is zero indexed, lower and upper are included.
	struct ExtractedFloatData
	{
		common::Box3I box;
		std::vector<float> data;
	};

	/// \brief Describes extracted data from a FITS file. box is zero indexed, lower and upper are included.
	struct ExtractedIntData
	{
		common::Box3I box;
		std::vector<int> data;
	};

	//// \brief Describes the min and max values of a set of data.
	struct MinMax
	{
		float min;
		float max;
	};

	auto isFitsFile(const std::filesystem::path& file) -> bool;

	/// \brief Extracts the bounding box of the data in the FITS file.
	/// \param fitsFilePath Path to the FITS file.
	/// \return A Box3I representing the bounding box of the data in the FITS file. Zero indexed, lower and upper are
	/// included.
	[[nodiscard]] auto extractBounds(const std::filesystem::path& fitsFilePath) -> common::Box3I;

	/// \brief Extracts some properties of the FITS file.
	/// \param fitsFilePath Path to the FITS file.
	/// \return Properties of the FITS file.
	auto getFitsProperties(const std::filesystem::path& fitsFilePath) -> b3d::tools::fits::FitsProperties;

	auto getFitsHeaderInfo(const std::filesystem::path& fitsFilePath) -> b3d::tools::fits::FitsHeaderInfo;

	/// \brief Reads the data from the FITS file as floats.
	/// \param fitsFilePath Path to the FITS file.
	/// \param searchBox Region to extract. Zero indexed, lower and upper are included. Use Box3I::maxBox() to extract everything.
	/// \return Extracted data.
	auto extractFloats(const std::filesystem::path& fitsFilePath, const common::Box3I& searchBox) -> ExtractedFloatData;

	/// \brief Reads the data from the FITS file as integers.
	/// \param fitsFilePath Path to the FITS file.
	/// \param searchBox Region to extract. Zero indexed, lower and upper are included. Use Box3I::maxBox() to extract
	/// everything. \return Extracted data.
	auto extractIntegers(const std::filesystem::path& fitsFilePath, const common::Box3I& searchBox) -> ExtractedIntData;

	/// \brief Create a new FITS file with the given data.
	/// \param fitsFilePath Path to the FITS file.
	/// \param boxSize Size of the data in each dimension.
	/// \param data Actual data to write.
	auto writeFitsFile(const std::filesystem::path& fitsFilePath, const common::Vec3I boxSize, const std::vector<float>& data)
		-> void;

	/// \brief Create a new FITS file with the given data.
	/// \param fitsFilePath Path to the FITS file.
	/// \param boxSize Size of the data in each dimension.
	/// \param data Actual data to write.
	auto writeFitsFile(const std::filesystem::path& fitsFilePath, const common::Vec3I boxSize, const std::vector<long>& data)
		-> void;

	/// \brief Apply a mask to the data and return the masked data as a new vector.
	/// \param data Input data to read from.
	/// \param mask Mask, true means keep the value, false means replace with maskedValue.
	/// \param maskedValue Value to replace the masked values.
	/// \return New vector with the masked data.
	auto applyMask(const std::vector<float>& data, const std::vector<bool>& mask, const float maskedValue)
		-> std::vector<float>;

	/// \brief Apply a mask to the data and return the masked data as a new vector.
	/// \param data Input data to read from.
	/// \param mask Mask, a mask value of 0 means replace with maskedValue.
	/// \param maskedValue Value to replace the masked values.
	/// \return New vector with the masked data.
	auto applyMask(const std::vector<float>& data, const std::vector<int>& mask, const float maskedValue)
		-> std::vector<float>;

	/// \brief Apply a mask to the data.
	/// \param data Input data which will be modified.
	/// \param mask Mask, a mask value of 0 means replace with maskedValue.
	/// \param maskedValue Value to replace the masked values.
	auto applyMaskInPlace(std::vector<float>& data, const std::vector<int>& mask, const float maskedValue) -> void;

	/// \brief Reads two fits files where the first file contains the data and the second file contains the mask. Applies the mask to the data and returns the masked data.
	/// \param fitsDataFilePath Path to the FITS file containing the data. Should contain float values.
	/// \param fitsMaskFilePath Path to the FITS file containing the mask. Should contain int values.
	/// \param maskedValue Value to replace the masked values.
	/// \return New vector with the masked data.
	auto applyMask(const std::filesystem::path& fitsDataFilePath, const std::filesystem::path& fitsMaskFilePath,
				   const float maskedValue) -> ExtractedFloatData;

	inline auto searchMinMax(const std::vector<float>& data) -> MinMax
	{
		const auto [min, max] = std::ranges::minmax_element(data);
		return { *min, *max };
	}

} // namespace b3d::tools::fits
