#include <nanovdb/util/CreateNanoGrid.h>
#include <util/IO.h>

#include "FitsTools.h"

#include "NanoTools.h"
#include "include/NanoTools_Internal.h"

using namespace b3d::common;
using namespace b3d::tools;

auto b3d::tools::nano::generateNanoVdb(const Vec3I boxSize, const float maskedValues, const float emptySpaceValue,
									   const std::vector<float>& data) -> nanovdb::GridHandle<>
{
	auto func = [&](const nanovdb::Coord& xyz)
	{
		const auto index = flattenIndex(boxSize, xyz.x(), xyz.y(), xyz.z());
		const auto v = data[index];
		return v == maskedValues ? emptySpaceValue : v;
	};

	const auto box =
		nanovdb::CoordBBox(nanovdb::Coord(0, 0, 0), nanovdb::Coord(boxSize.x - 1, boxSize.y - 1, boxSize.z - 1));
	nanovdb::build::Grid<float> grid(emptySpaceValue, "_nameless_", nanovdb::GridClass::FogVolume);
	grid(func, box);

	return nanovdb::createNanoGrid(grid);
}

auto b3d::tools::nano::generateNanoVdb(
	const Vec3I boxSize, const float emptySpaceValue,
	const std::function<float(const uint64_t x, const uint64_t y, const uint64_t z)>& f) -> nanovdb::GridHandle<>
{
	auto func = [&](const nanovdb::Coord& xyz) { return f(xyz.x(), xyz.y(), xyz.z()); };

	const auto box =
		nanovdb::CoordBBox(nanovdb::Coord(0, 0, 0), nanovdb::Coord(boxSize.x - 1, boxSize.y - 1, boxSize.z - 1));
	nanovdb::build::Grid<float> grid(emptySpaceValue, "_nameless_", nanovdb::GridClass::FogVolume);
	grid(func, box);

	return nanovdb::createNanoGrid(grid);
}

// TODO: Move to own library.
auto b3d::tools::nano::convertFitsWithMaskToNano(const std::filesystem::path& fitsDataFilePath,
												 const std::filesystem::path& fitsMaskFilePath,
												 const std::filesystem::path& destinationNanoVdbFilePath) -> NanoResult
{
	NanoResult result;

	const auto maskedValue = -100.0f;
	const auto maskedValues = b3d::tools::fits::applyMask(fitsDataFilePath, fitsMaskFilePath, maskedValue);

	const auto gridHandle =
		generateNanoVdb(maskedValues.box.size() + Vec3I{ 1, 1, 1 }, -100.0f, 0.0f, maskedValues.data);

	try
	{
		nanovdb::io::writeGrid(destinationNanoVdbFilePath.generic_string(), gridHandle,
							   nanovdb::io::Codec::NONE); // TODO: enable nanovdb::io::Codec::BLOSC
	}
	catch (...)
	{
		result.message = "Failed to write NanoVDB.";
		return result;
	}

	result.finished = true;
	result.resultFile = destinationNanoVdbFilePath.string();
	result.message = "Finished.";
	result.returnCode = 0;

	return result;
}

auto b3d::tools::nano::createNanoVdbWithExistingAndSubregion(
	const std::filesystem::path& sourceNanoVdbFilePath, const std::filesystem::path& originalFitsDataFilePath,
	const std::filesystem::path& originalFitsMaskFilePath, const std::filesystem::path& subRegionFitsMaskFilePath,
	const Vec3I& subRegionOffset, const std::filesystem::path& destinationNanoVdbFilePath) -> NanoResult
{
	NanoResult result;
	const auto subRegionMaskData = b3d::tools::fits::extractIntegers(subRegionFitsMaskFilePath, Box3I::maxBox());

	// Calculate Region and offset
	const auto subRegionInOriginalBox = Box3I{ subRegionOffset, subRegionOffset + subRegionMaskData.box.size() };

	auto originalFitsSubRegionData = b3d::tools::fits::extractFloats(originalFitsDataFilePath, subRegionInOriginalBox);
	const auto originalMaskSubRegionData =
		b3d::tools::fits::extractIntegers(originalFitsMaskFilePath, subRegionInOriginalBox);

	for (auto i = size_t{ 0 }; i < originalFitsSubRegionData.data.size(); ++i)
	{
		if ((subRegionMaskData.data[i] == 0) && (originalMaskSubRegionData.data[i] == 0))
		{
			originalFitsSubRegionData.data[i] = -100.0f;
		}
	}

	const auto originalFitsFileBounds = b3d::tools::fits::extractBounds(originalFitsDataFilePath);

	// Open the source NanoVDB
	// The grid of the source NanoVDB is most likely smaller than the grid of the original fits file.
	// Calculate offset of source NanoVDB in originalFitsFileBounds
	const auto sourceGridhandle = nanovdb::io::readGrid(sourceNanoVdbFilePath.generic_string());


	auto sourceGridAccessor = sourceGridhandle.grid<float>(0)->getAccessor();
	const auto sourceGridMetaData = sourceGridhandle.gridMetaData(0);

	// The sourceGridIndexBox is most likely smaller than
	// originalFitsFileBounds because empty space is not stored in the NanoVDB.
	const auto& sourceGridIndexBox = sourceGridMetaData->indexBBox();

	const auto sourceGridIndexBox3I =
		Box3I{ Vec3I{ sourceGridIndexBox.min()[0], sourceGridIndexBox.min()[1], sourceGridIndexBox.min()[2] },
			   Vec3I{ sourceGridIndexBox.max()[0], sourceGridIndexBox.max()[1], sourceGridIndexBox.max()[2] } };

	const auto subRegionAxisSize = subRegionMaskData.box.size() + Vec3I{ 1, 1, 1 };

	auto mapOriginalCoordinateToSubRegion = [=](const Vec3I& originalCoordinates)
	{ return originalCoordinates - subRegionOffset; };

	const auto func = [&](const uint64_t x, const uint64_t y, const uint64_t z) -> float
	{
		// If we're inside subregion, we use fits data from originalFitsSubRegionData.
		// If we're not inside subregion but inside sourceGridIndexBox , we use the source NanoVDB data.
		// Otherwise we return 0.0f for empty space.

		const auto position = Vec3I{ static_cast<int>(x), static_cast<int>(y), static_cast<int>(z) };

		if (subRegionInOriginalBox.lower.x <= position.x && subRegionInOriginalBox.lower.y <= position.y &&
			subRegionInOriginalBox.lower.z <= position.z && position.x <= subRegionInOriginalBox.upper.x &&
			position.y <= subRegionInOriginalBox.upper.y && position.z <= subRegionInOriginalBox.upper.z)
		{
			const auto subRegionCoordinate = mapOriginalCoordinateToSubRegion(position);

			const auto subRegionIdx = flattenIndex(subRegionAxisSize, subRegionCoordinate);
			return originalFitsSubRegionData.data[subRegionIdx] == -100.0f ?
				0.0f :
				originalFitsSubRegionData.data[subRegionIdx];
		}

		if (sourceGridIndexBox3I.lower.x <= position.x && sourceGridIndexBox3I.lower.y <= position.y &&
			sourceGridIndexBox3I.lower.z <= position.z && position.x <= sourceGridIndexBox3I.upper.x &&
			position.y <= sourceGridIndexBox3I.upper.y && position.z <= sourceGridIndexBox3I.upper.z)
		{
			return sourceGridAccessor.getValue(position.x, position.y, position.z);
		}

		return 0.0f;
	};

	auto gridHandle = generateNanoVdb(originalFitsFileBounds.size() + Vec3I{ 1, 1, 1 }, 0.0f, func);

	// Save nvdb
	try
	{
		nanovdb::io::writeGrid(destinationNanoVdbFilePath.string(), gridHandle,
							   nanovdb::io::Codec::NONE); // TODO: enable nanovdb::io::Codec::BLOSC
	}
	catch (...)
	{
		result.message = "Failed to write NanoVDB.";
		return result;
	}

	result.finished = true;
	result.resultFile = destinationNanoVdbFilePath.generic_string();
	result.message = "Finished.";
	result.returnCode = 0;

	return result;
}
