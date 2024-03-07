#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace b3d::renderer::nano
{
	enum class SelectMode
	{
		singleFile
	};

	class OpenFileDialog
	{
	public:
		OpenFileDialog(const SelectMode mode = SelectMode::singleFile, const std::vector<std::string>& filter = {})
			: mode_{ mode }, currentPath_{ std::filesystem::current_path() }, filter_{ filter }
		{
		}
		static auto open() -> void;
		auto gui() -> void;

		[[nodiscard]] inline auto getSelectedItems() const -> std::vector<std::filesystem::path>
		{
			return selectedItems_;
		}
	private:
		std::filesystem::path currentPath_{};
		std::filesystem::path selectedPath_{};

		SelectMode mode_{SelectMode::singleFile};
		std::vector<std::string> filter_{};


		std::vector<std::filesystem::path> selectedItems_{};
	};
}
