#pragma once

#include <filesystem>
#include <future>

namespace b3d::tools::project
{
	class ServerClient;

	namespace catalog
	{
		class FileCatalog;	
	}
}

namespace b3d::tools::project
{
	class ServerFileProvider
	{
	public:
		ServerFileProvider(std::filesystem::path dataRootPath, ServerClient& serverClient);
		~ServerFileProvider();

		auto isFileAvailable(const std::string& fileUUID) const -> bool;
		auto getFilePath(const std::string& fileUUID) const -> std::optional<std::filesystem::path>;

		auto loadFileFromServerAsync(const std::string& fileUUID, bool reloadFile) const -> std::future<bool>;

	private:
		std::unique_ptr<catalog::FileCatalog> fileCatalog_;
		ServerClient& serverClient_;

		auto loadFileAndAddToCatalog(const std::string& fileUUID) const -> bool;

		auto loadFileFromServer(const std::string& fileUUID) const -> std::optional<std::filesystem::path>;
	};
} // namespace b3d::tools::project
