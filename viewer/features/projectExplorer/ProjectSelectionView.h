#pragma once

#include "framework/ModalViewBase.h"
class ProjectSelectionView final : public ModalViewBase
{
public:
	explicit ProjectSelectionView(ApplicationContext& applicationContext);

	auto onDraw() -> void override;
};
