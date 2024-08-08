#pragma once

#include <string>

struct ConnectionData
{
	std::string port;
	std::string ip;
};

class ServerConnect
{
private:
	bool isLocalhost_{ true };

public:
	[[nodiscard]] auto checkHealth() const -> bool;
};
