#pragma once
class ApplicationContext;

/// \brief UpdatableComponentBase objects get updated every frame when it's added to ApplicationContext. Allows access to the ApplicationContext.
class UpdatableComponentBase
{
public:
	explicit UpdatableComponentBase(ApplicationContext& applicationContext);

	virtual ~UpdatableComponentBase()
	{
	}

	virtual auto update() -> void = 0;

protected:
	ApplicationContext* applicationContext_{};
};
