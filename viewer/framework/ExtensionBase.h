#pragma once

class ApplicationContext;

class ExtensionBase
{
public:
	explicit ExtensionBase(ApplicationContext& applicationContext) : appContext_{ &applicationContext }
	{
	}

	virtual ~ExtensionBase(){};

public:
	virtual auto initializeResources() -> void = 0;
	virtual auto deinitializeResources() -> void = 0;

	ApplicationContext* appContext_;
};
