#pragma once

class Pass
{
public:
	virtual auto execute() const -> void = 0;
	virtual ~Pass()
	{
	}
};
