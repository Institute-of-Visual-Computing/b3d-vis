#pragma once

#include <SofiaParams.h>
#include "framework/ModalViewBase.h"

class SofiaParameterSummaryView final : public ModalViewBase
{
public:
	SofiaParameterSummaryView(ApplicationContext& applicationContext);

	auto onDraw() -> void override;

private:
	b3d::tools::sofia::SofiaParams params_{};
public:
	auto setSofiaParams(const b3d::tools::sofia::SofiaParams& params) -> void
	{
		params_ = params;
	}
};
