#include <uuid.h>

#include "Request.h"

static uuids::uuid_name_generator gNameGenerator{ uuids::uuid::from_string("123456789-abcdef-123456789-abcdef-12").value() };

auto b3d::tools::project::Request::createUUID() const -> std::string
{
	return uuids::to_string(gNameGenerator(sofiaParameters.buildCliArgumentsAsString()));
}
