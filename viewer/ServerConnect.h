#pragma once

class ServerConnect
{
public:
	[[nodiscard]] auto IsConnected() const -> bool
	{
		return isConnected_;
	}

private:
	bool isConnected_{ false };
	bool isLocalhost_{ false };
};
