#include "FileCatalog.h"
#include "ServerClient.h"

#include "ServerFileProvider.h"


b3d::tools::project::ServerFileProvider::ServerFileProvider(std::filesystem::path dataRootPath,
															ServerClient& serverClient)
	: serverClient_{ serverClient }
{
	fileCatalog_ = std::make_unique <catalog::FileCatalog>(catalog::FileCatalog::createOrLoadCatalogFromPathes(dataRootPath, "serverCache"));
}

b3d::tools::project::ServerFileProvider::~ServerFileProvider() = default;

auto b3d::tools::project::ServerFileProvider::isFileAvailable(const std::string& fileUUID) const -> bool
{
	return fileCatalog_->contains(fileUUID);
}

auto b3d::tools::project::ServerFileProvider::getFilePath(
	const std::string& fileUUID) const -> std::optional<std::filesystem::path>
{
	if (fileCatalog_->contains(fileUUID))
	{
		return fileCatalog_->getFilePathAbsolute(fileUUID);
	}
	return std::nullopt;
}

auto b3d::tools::project::ServerFileProvider::addLocalFile(const std::filesystem::path& filePath) const -> std::string
{
	return fileCatalog_->addFilePathAbsolute(filePath, false);
}

auto b3d::tools::project::ServerFileProvider::loadFileFromServerAsync(const std::string& fileUUID,
                                                                      bool reloadFile) const
	-> std::future<bool>
{
	if (fileCatalog_->contains(fileUUID))
	{
		if (!reloadFile)
		{
			std::promise<bool> promise;
			auto future = promise.get_future();
			promise.set_value(true);
			return future;
		}
	}
	
	fileCatalog_->removeMapping(fileUUID);
	// return std::async(std::launch::async, []() { return true; });

	return std::async(std::launch::async, [this, fileUUID]() { return loadFileAndAddToCatalog(fileUUID); });
	// return std::async(std::launch::async, this->loadFileAndAddToCatalog, fileUUID);
}

auto b3d::tools::project::ServerFileProvider::loadFileAndAddToCatalog(const std::string& fileUUID) const -> bool
{
	const auto newFilePath = loadFileFromServer(fileUUID);
	if (newFilePath.has_value())
	{
		fileCatalog_->addFilePathAbsoluteWithUUID(newFilePath.value(), fileUUID);
		return true;
	}
	return false;
}

auto b3d::tools::project::ServerFileProvider::loadFileFromServer(
	const std::string& fileUUID) const -> std::optional<std::filesystem::path>
{
	auto downloadFuture = serverClient_.downloadFileAsync(fileUUID, fileCatalog_->getDataPathAbsolute());
	downloadFuture.wait();
	const auto downloadPath = downloadFuture.get();
	if (!downloadPath.empty())
	{
		return downloadPath;
	}
	return std::nullopt;
}
