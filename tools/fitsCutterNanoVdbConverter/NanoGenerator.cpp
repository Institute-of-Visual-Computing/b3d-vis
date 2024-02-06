#include "FitsHelper.h"

#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Primitives.h>

auto generateNanoVdb(const std::string& file, const Vec3I boxSize, float maskedValues, float emptySpaceValue,
					 const std::vector<float>& data) -> void
{
	auto func = [&](const nanovdb::Coord& ijk)
	{
		const auto i = ijk.x();
		const auto j = ijk.y();
		const auto k = ijk.z();
		const auto index = static_cast<unsigned long long>(boxSize.x) * static_cast<unsigned long long>(boxSize.y) * k +
			static_cast<unsigned long long>(boxSize.x) * j + i;
		const auto v = data[index];
		return v == maskedValues ? emptySpaceValue : v;
	};

	const auto box =
		nanovdb::CoordBBox(nanovdb::Coord(0, 0, 0), nanovdb::Coord(boxSize.x - 1, boxSize.y - 1, boxSize.z - 1));
	nanovdb::build::Grid<float> grid(emptySpaceValue, "funny", nanovdb::GridClass::FogVolume);
	grid(func, box);

	const auto gridHandle = nanovdb::createNanoGrid(grid);
	std::println(std::cout, "NanoVdb buffer size: {}bytes", gridHandle.size());

	nanovdb::io::writeGrid(file, gridHandle,
						   nanovdb::io::Codec::NONE); // TODO: enable nanovdb::io::Codec::BLOSC
}
