#pragma once
#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <string>
#include <vector>


namespace cutterParser
{

	struct Box
	{
		std::array<float, 3> min;
		std::array<float, 3> max;
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Box, min, max);
	};

	struct TreeNode
	{
		std::string nanoVdbFile{};
		std::size_t nanoVdbBufferSize{};
		Box aabb{};
		std::vector<TreeNode> children{};
		float minValue;
		float maxValue;
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(TreeNode, nanoVdbFile, nanoVdbBufferSize, aabb, children, minValue, maxValue)
	};

	using json = nlohmann::json;

	auto load(const std::filesystem::path& file) -> std::vector<TreeNode>;
	auto store(const std::filesystem::path& file, const std::vector<TreeNode>& trees) -> void;
} // namespace cutterParser
