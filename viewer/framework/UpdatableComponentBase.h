#pragma once
class ApplicationContext;

class UpdatableComponentBase
{
public:
	UpdatableComponentBase(ApplicationContext& applicationContext);

	virtual ~UpdatableComponentBase()
	{
	}

	virtual auto update() -> void = 0;

protected:
	ApplicationContext* applicationContext_{};
};
