#pragma once

#include "../DockableWindowViewBase.h"

class ServerConnectView : public DockableWindowViewBase
{
protected:
	auto onDraw() -> void override;
};
