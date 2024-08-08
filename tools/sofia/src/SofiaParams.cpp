#include "SofiaParams.h"

b3d::tools::sofia::SofiaParams::SofiaParams(std::initializer_list<std::pair<const std::string, std::string>> params)
{
	for (const auto& param : params)
	{
		if (isKeyValid(param.first))
		{
			params_.insert(param);
		}
	}
}

b3d::tools::sofia::SofiaParams::SofiaParams(const std::map<std::string, std::string> keyValueMap) : params_(keyValueMap)
{
	for (const auto& key : keyValueMap | std::views::keys)
	{
		if (!isKeyValid(key))
		{
			params_.erase(key);
		}
	}
}

auto b3d::tools::sofia::SofiaParams::isKeyValid(const std::string& key) -> bool
{
	return DEFAULT_PARAMS.contains(key);
}

auto b3d::tools::sofia::SofiaParams::setOrReplace(const std::string& key, const std::string& value) -> bool
{
	if (!isKeyValid(key))
	{
		return false;
	}
    params_.erase(key) ;
	return params_.insert({key, value}).second;
}

auto b3d::tools::sofia::SofiaParams::getStringValue(const std::string& key) -> std::optional<std::string>
{
	return params_.contains(key) ? std::make_optional(params_.at(key)) : std::nullopt;
}

auto b3d::tools::sofia::SofiaParams::containsKey(const std::string& key) const -> bool
{
	return params_.contains(key);
}

auto b3d::tools::sofia::SofiaParams::getKeyValueCliArgument(const std::string& key) const -> std::string
{
	const auto& it = params_.find(key);
	if (it == params_.end())
	{
		return "";
	}

	return concatCLI(it->first, it->second);
}

auto b3d::tools::sofia::SofiaParams::buildCliArguments(bool useDefaultIfKeyMissing) const -> std::vector<std::string>
{
	std::vector<std::string> cliArgs = {};
	// TODO: ..par-file?

	// When .par-file is used do not use the default values. Just use the values from the map.

	for (const auto& [key, defaultValue] : DEFAULT_PARAMS)
	{
		const auto& it = params_.find(key);
		if (it != params_.end() && !it->second.empty())
		{
			cliArgs.emplace_back(concatCLI(it->first, it->second));
		}
		else if (useDefaultIfKeyMissing && !defaultValue.empty())
		{
			cliArgs.emplace_back(concatCLI(key, defaultValue));
		}
	}
	return cliArgs;
}

auto b3d::tools::sofia::SofiaParams::buildCliArgumentsAsString(bool useDefaultIfKeyMissing) const -> std::string
{
	std::string cliArgs = "";
	// TODO: ..par-file?

	// When .par-file is used do not use the default values. Just use the values from the map.

	for (const auto& [key, defaultValue] : DEFAULT_PARAMS)
	{
		const auto& it = params_.find(key);
		if (it != params_.end() && !it->second.empty())
		{
			cliArgs += concatCLI(it->first, it->second) + " ";
		}
		else if (useDefaultIfKeyMissing && !defaultValue.empty())
		{
			cliArgs += concatCLI(key, defaultValue) + " ";
		}
	}
	return cliArgs;
}
