#pragma once

#include <functional>

#include <nanovdb/util/GridHandle.h>

#include "Vec.h"

#include "NanoResult.h"

namespace b3d::tools::nano
{
	/// \brief Generate a NanoVDB FogVolume grid from a 3D data array.
	/// \param boxSize Size of the 3D data array.
	/// \param maskedValues Value of the masked values.
	/// \param emptySpaceValue Value which represents empty space, maskedValues will be replaced with this value.
	/// \param data 3d data array.
	/// \return GridHandle to the generated NanoVDB grid.
	auto generateNanoVdb(const b3d::common::Vec3I boxSize, const float maskedValues, const float emptySpaceValue,
	                     const std::vector<float>& data) -> nanovdb::GridHandle<>;

	/// \brief Generate a NanoVDB FogVolume grid from a 3D function.
	/// \param boxSize Size of the 3D function.
	/// \param emptySpaceValue Value which represents empty space.
	/// \param f 3D function, which gets called.
	/// \return GridHandle to the generated NanoVDB grid.
	auto generateNanoVdb(const b3d::common::Vec3I boxSize, const float emptySpaceValue,
	                     const std::function<float(const uint64_t i, const uint64_t j, const uint64_t k)>& f)
		-> nanovdb::GridHandle<>;

	/// \brief Convert a FITS file with a mask to a NanoVDB file.
	/// \param fitsDataFilePath Path to the FITS file.
	/// \param fitsMaskFilePath Path to the FITS mask file. Same dimensions as file at fitsDataFilePath.
	/// \param destinationNanoVdbFilePath Path where the NanoVDB file should be saved.
	/// \return NanoResult for the volume data in the nvdb
	///	TODO: Move to own library.
	auto convertFitsWithMaskToNano(const std::filesystem::path& fitsDataFilePath,
	                               const std::filesystem::path& fitsMaskFilePath,
								   const std::filesystem::path& destinationNanoVdbFilePath)
		-> NanoResult;

	/// \brief Convert a FITS file to a NanoVDB file without a mask file.
	///	\param fitsDataFilePath Path to the FITS file.
	/// \param destinationNanoVdbFilePath Path where the NanoVDB file should be saved.
	/// \return NanoResult for the volume data in the nvdb
	auto convertFitsToNano(const std::filesystem::path& fitsDataFilePath,
						   const std::filesystem::path& destinationNanoVdbFilePath) -> NanoResult;

	/// \brief Creates a new NanoVDB based on an existing NanoVDB. The content in the subregion will be altered based on the merged masks
	/// \param sourceNanoVdbFilePath Path to the source NanoVDB file.
	/// \param originalFitsDataFilePath Path to the original FITS data file.
	/// \param originalFitsMaskFilePath Path to the original FITS mask file. Same dimensions as file at originalFitsDataFilePath.
	/// \param subRegionFitsMaskFilePath Path to the subregion FITS mask file.
	/// \param subRegionOffset Offset of the subregion in the original FITS data.
	/// \param destinationNanoVdbFilePath Path where the new NanoVDB file should be saved.
	/// \return true if the conversion was successful, false otherwise.
	auto createNanoVdbWithExistingAndSubregion(const std::filesystem::path& sourceNanoVdbFilePath,
	                                           const std::filesystem::path& originalFitsDataFilePath,
	                                           const std::filesystem::path& originalFitsMaskFilePath,
	                                           const std::filesystem::path& subRegionFitsMaskFilePath,
											   const b3d::common::Vec3I& subRegionOffset,	
	                                           const std::filesystem::path& destinationNanoVdbFilePath)
		-> NanoResult;
	
} // namespace b3d::tools::nano
