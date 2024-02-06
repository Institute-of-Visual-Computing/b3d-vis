#include "FitsHelper.h"

#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>

auto generateNanoVdb(const Vec3I boxSize, float maskedValues, float emptySpaceValue,
					 const std::vector<float>& data) -> nanovdb::GridHandle<>
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
	nanovdb::build::Grid<float> grid(emptySpaceValue, "_nameless_", nanovdb::GridClass::FogVolume);
	grid(func, box);

	return nanovdb::createNanoGrid(grid);
}
