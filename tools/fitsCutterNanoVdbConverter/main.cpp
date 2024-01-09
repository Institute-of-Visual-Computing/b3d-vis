#include <cassert>
#include <cfitsio/fitsio.h>
#include <filesystem>
#include <iostream>

auto fitsDeleter(fitsfile* file) -> void
{
	auto status = int{};
	ffclos(file, &status);
	assert(status == 0);
};

using unique_fitsfile = std::unique_ptr<fitsfile, decltype(&fitsDeleter)>;

auto main() -> int
{

	const auto fitsFilePath = std::filesystem::path{ "D:/datacubes/testDataSet/n4565.fits" };
	const auto catalogFilePath = std::filesystem::path{ "D:/datacubes/testDataSet/n4565_catalog.fits" };
	const auto maskFilePath = std::filesystem::path{ "D:/datacubes/testDataSet/n4565_mask.fits" };


	
	fitsfile* fitsFilePtr{ nullptr };
	auto fitsError = int{};
	ffopen(&fitsFilePtr, fitsFilePath.generic_string().c_str(), READONLY, &fitsError);
	assert(fitsError == 0);
	
	auto fitsFile = unique_fitsfile(fitsFilePtr, &fitsDeleter);

	int axisCount;
    int imgType;
    long axis[3];
    fits_get_img_param(fitsFile.get(), 3, &imgType, &axisCount, &axis[0], &fitsError);


	//fits_read_subset(fitsFile.get(), type, )

	// fits_get_img_param()
	std::cout << "hallo world!" << std::endl;
	return EXIT_SUCCESS;
}
