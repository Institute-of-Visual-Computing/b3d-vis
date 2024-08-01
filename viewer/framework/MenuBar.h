#pragma once

class ApplicationContext;

class MenuBar final
{
public:
	explicit MenuBar(ApplicationContext& applicationContext);
	auto draw() const -> void;
	~MenuBar()
	{
	}

private:
	ApplicationContext* applicationContext_;
};
