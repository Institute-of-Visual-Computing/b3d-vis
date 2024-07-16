#include "uuid.h"


namespace b3d::tools::projectexplorer
{
	uuids::uuid_name_generator gNameGenerator{
		uuids::uuid::from_string("123456789-abcdef-123456789-abcdef-12").value()
	};
}
