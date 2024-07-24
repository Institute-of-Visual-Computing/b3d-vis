#include "FitsTools.h"

#include <algorithm>
#include <cassert>
#include <execution>
#include <format>
#include <iostream>

#include <cfitsio/fitsio.h>

#include "FitsCommon.h"
#include "FitsHelper.h"

auto b3d::tools::fitstools::getFitsFileInfos(const std::filesystem::path& file, const uint8_t hduIndex)
	-> common::fits::FitsProperties
{
	common::fits::FitsProperties fitsProps;

	fitsfile* fitsFilePtr{ nullptr };
	auto fitsError = int{};
	ffopen(&fitsFilePtr, file.generic_string().c_str(), READONLY, &fitsError);
	assert(fitsError == 0);

	const auto fitsFile = UniqueFitsfile(fitsFilePtr, &fitsDeleter);

	
	fits_get_img_param(fitsFile.get(), 3, &fitsProps.imgType, &fitsProps.axisCount, nullptr, &fitsError);
	assert(fitsProps.axisCount > 0);
	fitsProps.axisDimensions.resize(fitsProps.axisCount);
	fitsProps.axisTypes.resize(fitsProps.axisCount);
	fits_get_img_param(fitsFile.get(), 3, &fitsProps.imgType, &fitsProps.axisCount, fitsProps.axisDimensions.data(),
					   &fitsError);

	auto hduCount = 0;
	fits_get_num_hdus(fitsFile.get(), &hduCount, &fitsError);
	assert(fitsError == 0);

	assert(hduCount > 0);


	assert(hduIndex < hduCount);
	fits_movabs_hdu(fitsFile.get(), hduIndex + 1, nullptr, &fitsError);
	assert(fitsError == 0);

	// Read ctype
	{
		std::vector<char> keyValue;
		// Fits TSTRING is 68 chars + \0
		keyValue.resize(69);
		for (auto i = 0; i < fitsProps.axisCount; i++)
		{
			auto retVal = fits_read_key(fitsFile.get(), TSTRING, std::format("CTYPE{}", i + 1).c_str(), keyValue.data(),
										NULL, &fitsError);
			if (fitsError != 0)
			{
				std::cout << "CTYPE not found"
						  << "\n";
				break;
			}
			fitsProps.axisTypes[i] = std::string{ keyValue.data() };
		}
	}

	return fitsProps;
}
