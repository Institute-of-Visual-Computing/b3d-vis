#pragma once

class ApplicationContext;

class MenuBar
{
public:
	MenuBar(ApplicationContext& applicationContext);
	auto draw() -> void;
	virtual ~MenuBar(){}

		private:
	ApplicationContext* applicationContext_;
};
