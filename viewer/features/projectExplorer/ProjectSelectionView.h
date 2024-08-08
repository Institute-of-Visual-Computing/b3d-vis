#pragma once

#include "framework/ModalViewBase.h"
class ProjectSelectionView : public ModalViewBase
{
public:
	ProjectSelectionView(ApplicationContext& applicationContext);

	auto onDraw() -> void override;
};
