#pragma once

#include "framework/DockableWindowViewBase.h"

class ServerConnectView : public DockableWindowViewBase
{
protected:
	auto onDraw() -> void override;
};
