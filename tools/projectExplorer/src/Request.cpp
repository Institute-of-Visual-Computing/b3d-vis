#include "Request.h"

#include "uuid.h"

#include "Common.h"

auto b3d::tools::projectexplorer::Request::createUUID() const -> std::string
{
	const auto s = std::accumulate(sofiaSearchParameters.begin(), sofiaSearchParameters.end(), std::string{});
	return to_string(gNameGenerator(s));
}
