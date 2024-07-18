#pragma once

#include "FontCollection.h"

class ApplicationContext final
{
public:
	[[nodiscard]] auto getFontCollection() -> FontCollection&
	{
		return fonts_;
	}

private:
	FontCollection fonts_{};
};
