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
		Box aabb{};
		std::vector<TreeNode> children{};

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(TreeNode, nanoVdbFile, aabb, children)
	};

	using json = nlohmann::json;

	auto load(const std::filesystem::path& file) -> std::vector<TreeNode>;
	auto store(const std::filesystem::path& file, const std::vector<TreeNode>& trees) -> void;
} // namespace cutterParser
