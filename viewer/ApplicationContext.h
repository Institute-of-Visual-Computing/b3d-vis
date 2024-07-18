#pragma once

#include "FontCollection.h"


class GLFWwindow;

class ApplicationContext final
{
public:
	[[nodiscard]] auto getFontCollection() -> FontCollection&
	{
		return fonts_;
	}

	GLFWwindow* mainWindowHandle_{};

private:
	FontCollection fonts_{};
};
