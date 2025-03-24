#include "FitsHelper.h"


#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

auto generateNanoVdb(const Vec3I boxSize, const float maskedValues, const float emptySpaceValue,
					 const std::vector<float>& data) -> nanovdb::GridHandle<>
{
	auto func = [&](const nanovdb::Coord& ijk) -> float
	{
		const auto i = ijk.x();
		const auto j = ijk.y();
		const auto k = ijk.z();
		const auto index = static_cast<unsigned long long>(boxSize.x) * static_cast<unsigned long long>(boxSize.y) * k +
			static_cast<unsigned long long>(boxSize.x) * j + i;
		const auto v = data[index];
		return v == maskedValues or isnan(v) ? emptySpaceValue : v;
	};

	const auto box =
		nanovdb::CoordBBox(nanovdb::Coord(0, 0, 0), nanovdb::Coord(boxSize.x - 1, boxSize.y - 1, boxSize.z - 1));
	nanovdb::tools::build::Grid<float> grid(emptySpaceValue, "_nameless_", nanovdb::GridClass::FogVolume);
	grid(func, box);

	return nanovdb::tools::createNanoGrid(grid);
}

auto generateNanoVdb(const Vec3I boxSize, const float emptySpaceValue,
					 const std::function<float(const uint64_t i, const uint64_t j, const uint64_t k)>& f)-> nanovdb::GridHandle<>
{
	auto func = [&](const nanovdb::Coord& ijk)
	{
		const auto i = ijk.x();
		const auto j = ijk.y();
		const auto k = ijk.z();
		return f(i,j,k);
	};
	const auto box =
		nanovdb::CoordBBox(nanovdb::Coord(0, 0, 0), nanovdb::Coord(boxSize.x - 1, boxSize.y - 1, boxSize.z - 1));
	nanovdb::tools::build::Grid<float> grid(emptySpaceValue, "_nameless_", nanovdb::GridClass::FogVolume);
	grid(func, box);

	return nanovdb::tools::createNanoGrid(grid);
}
