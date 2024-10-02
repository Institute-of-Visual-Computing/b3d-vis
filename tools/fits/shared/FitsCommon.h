#pragma once

#include <string>
#include <vector>

#ifdef B3D_USE_NLOHMANN_JSON
	#include <nlohmann/json.hpp>
#endif

namespace b3d::tools::fits
{
	/// \brief Copied from cfitsio.h. Not to mistaken with data length in bytes.
	enum class FitsImageType : int
	{
		UNKNOWN = 0,
		BYTE = 8,
		SHORT = 16,
		INT = 32,
		LONG = 64,
		FLOAT = -32,
		DOUBLE = -64
	};

	/// \brief Copied from cfitsio.h. Not to mistaken with data length in bytes.
	enum class FitsDataTypes : int
	{
		UNKNOWN = 0,
		SBYTE = 11,
		USHORT = 20,
		SHORT = 21,
		UINT = 30,
		INT = 31,
		LONG = 41,
		ULONG = 40,
		FLOAT = 42,
		LONGLONG = 81,
		DOUBLE = 82
	};

	//// \brief Common Properties of a FITS file used in the library.
	struct FitsProperties
	{
		int axisCount;
		FitsImageType imgType;
		std::vector<long> axisDimensions;
		std::vector<std::string> axisTypes;
	};

	#ifdef B3D_USE_NLOHMANN_JSON
		NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FitsProperties, axisCount, imgType, axisDimensions, axisTypes);
	#endif

} // namespace b3d::tools::fits
