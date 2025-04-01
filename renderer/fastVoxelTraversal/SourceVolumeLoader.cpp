#include "SourceVolumeLoader.h"

#include <map>

#include <cfitsio/fitsio.h>
#include <owl/common/math/box.h>
#include <tinyxml2.h>

#include "FastVoxelTraversalSharedStructs.h"

auto SourceVolumeLoader::extractSourceRegionsFromCatalogueXML(const std::string& filePath,
															  std::vector<SourceRegion>& sourceBoxes) -> size_t
{
	struct SourceRegionLocation
	{
		std::map<std::string, int> boxMinMax = { { "x_min", -1 }, { "x_max", -1 }, { "y_min", -1 },
												 { "y_max", -1 }, { "z_min", -1 }, { "z_max", -1 } };

		owl::box3i toBox3i()
		{
			return { { boxMinMax["x_min"], boxMinMax["y_min"], boxMinMax["z_min"] },
					 { boxMinMax["x_max"], boxMinMax["y_max"], boxMinMax["z_max"] } };
		}

		owl::box3f toBox3f()
		{
			const auto x_min = static_cast<float>(boxMinMax["x_min"]);
			const auto x_max = static_cast<float>(boxMinMax["x_max"]);
			const auto y_min = static_cast<float>(boxMinMax["y_min"]);
			const auto y_max = static_cast<float>(boxMinMax["y_max"]);
			const auto z_min = static_cast<float>(boxMinMax["z_min"]);
			const auto z_max = static_cast<float>(boxMinMax["z_max"]);
			return { { x_min, y_min, z_min }, { x_max, y_max, z_max } };
		}
	};

	size_t voxels = 0;
	tinyxml2::XMLDocument doc;
	auto xmlError = doc.LoadFile(filePath.c_str());
	if (xmlError != tinyxml2::XML_SUCCESS)
	{
		return -1;
	}

	//<VOTABLE>
	//  <RESOURCE>
	//    ...
	//    ...
	//    <TABLE>
	//      <FIELD name="fieldname1" datatype="datatype" .... />
	//      <FIELD name="fieldname2" datatype="datatype" .... />
	//      ...
	//      <FIELD name="fieldnameX" datatype="datatype" .... />
	//      <DATA>
	//        <TABLEDATA>
	//          <TR>
	//            <TD> value for fieldname1 </TD>
	//            <TD> value for fieldname2 </TD>
	//            ...
	//            <TD> value for fieldnameX </TD>
	//          </TR>
	//          <TR>
	//            ...
	//          </TR>
	//            ...
	//          <TR>
	//          </TR>
	//        </TABLEDATA>
	//      </DATA>

	//    </TABLE>
	//  </RESOURCE>
	//</VOTABLE>

	auto tableElement = doc.FirstChildElement("VOTABLE")->FirstChildElement("RESOURCE")->FirstChildElement("TABLE");
	auto tableChildNameEntrys = tableElement->FirstChild();

	auto dataEntry = tableElement->LastChildElement("DATA")->FirstChildElement("TABLEDATA")->FirstChild();

	for (; dataEntry != nullptr; dataEntry = dataEntry->NextSibling())
	{
		SourceRegionLocation srl{};
		auto tdEntry = dataEntry->FirstChild();
		for (auto currTableChildNameEntry = tableChildNameEntrys; currTableChildNameEntry != nullptr;
			 currTableChildNameEntry = currTableChildNameEntry->NextSibling())
		{
			auto tableChildElement = currTableChildNameEntry->ToElement();

			if (tableChildElement == nullptr || !tableChildElement->NoChildren())
			{
				break;
			}
			std::string attributeName = tableChildElement->Attribute("name");
			if (!attributeName.empty() && srl.boxMinMax.find(attributeName) != srl.boxMinMax.end())
			{
				auto entryValue = tdEntry->ToElement()->IntText();
				srl.boxMinMax[attributeName] = entryValue;
			}
			tdEntry = tdEntry->NextSibling();
		}
		SourceRegion sr;
		sr.gridSourceBox = srl.toBox3i();
		sr.bufferOffset = voxels;
		voxels += sr.gridSourceBox.volume();
		sr.sourceBoxNormalized = srl.toBox3f();
		sourceBoxes.push_back(sr);
	}

	return voxels;
}

auto SourceVolumeLoader::loadDataForSources(const std::string& filePath, std::vector<SourceRegion>& sourceBoxes,
											std::vector<float>& dataBuffer) -> owl::vec3i
{
	long firstPx[3];
	long lastPx[3];
	long inc[3] = { 1, 1, 1 };
	size_t nextDataIdx = 0;

	fitsfile* fitsFile;
	int fitsError;
	ffopen(&fitsFile, filePath.c_str(), READONLY, &fitsError);
	assert(fitsError == 0);

	int axisCount;
	int imgType;
	long axis[3];
	fits_get_img_param(fitsFile, 3, &imgType, &axisCount, &axis[0], &fitsError);
	assert(fitsError == 0);
	assert(axisCount == 3);
	assert(imgType == FLOAT_IMG);


	const owl::box3i dataCubeVolume({ 0, 0, 0 }, { axis[0] - 1, axis[1] - 1, axis[2] - 1 });
	vec3f dims = { static_cast<float>(axis[0] - 1), static_cast<float>(axis[1] - 1), static_cast<float>(axis[2] - 1) };
	float nan = NAN;

	for (auto& sourceBox : sourceBoxes)
	{
		sourceBox.bufferOffset = nextDataIdx;
		firstPx[0] = sourceBox.gridSourceBox.lower.x + 1;
		firstPx[1] = sourceBox.gridSourceBox.lower.y + 1;
		firstPx[2] = sourceBox.gridSourceBox.lower.z + 1;

		lastPx[0] = sourceBox.gridSourceBox.upper.x;
		lastPx[1] = sourceBox.gridSourceBox.upper.y;
		lastPx[2] = sourceBox.gridSourceBox.upper.z;

		fits_read_subset(fitsFile, TFLOAT, firstPx, lastPx, inc, &nan, dataBuffer.data() + nextDataIdx, 0, &fitsError);
		assert(fitsError == 0);
		nextDataIdx += sourceBox.gridSourceBox.volume();
		sourceBox.sourceBoxNormalized.lower =
			vec3f{ sourceBox.sourceBoxNormalized.lower.x / dims.x, sourceBox.sourceBoxNormalized.lower.y / dims.y,
				   sourceBox.sourceBoxNormalized.lower.z / dims.z } -
			vec3f{ .5f };
		sourceBox.sourceBoxNormalized.upper =
			vec3f{ sourceBox.sourceBoxNormalized.upper.x / dims.x, sourceBox.sourceBoxNormalized.upper.y / dims.y,
				   sourceBox.sourceBoxNormalized.upper.z / dims.z } -
			vec3f{ .5f };
	}

	ffclos(fitsFile, &fitsError);
	return { axis[0], axis[1], axis[2] };
}
