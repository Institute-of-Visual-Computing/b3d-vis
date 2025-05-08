#pragma once

class Pass
{
public:
	virtual auto execute() -> void = 0;
	virtual ~Pass()
	{
	}
};
