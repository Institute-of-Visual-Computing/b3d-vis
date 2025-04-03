#include <filesystem>
#include <future>
#include <string>
#include <vector>

#include <args.hxx>
#include <boost/process.hpp>
#include <httplib.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/ConsoleInitializer.h>
#include <plog/Log.h>
#include <uuid.h>

#include <FitsTools.h>
#include <NanoTools.h>
#include <SofiaNanoPipeline.h>
#include <SofiaProcessRunner.h>
#include <TimeStamp.h>

#include "src/Internals.h"
#include "src/ProjectProvider.h"

using namespace b3d::tools::projectServer;

// https://stackoverflow.com/questions/2989810/which-cross-platform-preprocessor-defines-win32-or-win32-or-win32
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix))
std::filesystem::path sofiaPath = boost::process::search_path("sofia").generic_string();
#else
std::filesystem::path sofiaPath = boost::process::search_path("sofia.exe").generic_string();

#endif


auto commonRootPath = boost::process::filesystem::path("");

std::unique_ptr<std::future<InternalRequest>> currentRequest{ nullptr };
std::unique_ptr<ProjectProvider> projectProvider{ nullptr };
std::unique_ptr<b3d::tools::sofia::SofiaProcessRunner> sofiaProcessRunner{ nullptr };

std::mutex currentRequestMutex;

auto processCurrentRequest() -> void
{
	std::lock_guard lock(currentRequestMutex);
	LOG_INFO << "Processing current request";

	using namespace std::chrono_literals;
	if (currentRequest == nullptr)
	{
		return;
	}

	if (!currentRequest->valid())
	{
		return;
	}

	const auto waitResult = currentRequest->wait_for(0s);
	if (waitResult != std::future_status::ready)
	{
		LOG_INFO << "Current request running";
		return;
	}

	LOG_INFO << "Current request done";
	// Save request and modify paths to UUIDs

	auto req = currentRequest->get();
	auto& projectCatalog = projectProvider->getCatalog(req.projectUUID);
	if (req.userRequest.result.sofiaResult.wasSuccess())
	{

		req.userRequest.result.sofiaResult.resultFile =
			projectCatalog.addFilePathAbsolute(req.userRequest.result.sofiaResult.resultFile);
	}

	if (req.userRequest.result.nanoResult.wasSuccess())
	{
		if (req.userRequest.result.sofiaResult.returnCode == 8)
		{
			req.userRequest.result.sofiaResult.resultFile =
				projectCatalog.addFilePathAbsolute(req.userRequest.result.sofiaResult.resultFile);
		}

		req.userRequest.result.nanoResult.resultFile =
			projectCatalog.addFilePathAbsolute(req.userRequest.result.nanoResult.resultFile);
	}

	LOG_INFO << "Saving current request";
	projectProvider->getProject(req.projectUUID).requests.emplace_back(req.userRequest);
	projectProvider->saveProject(req.projectUUID);
	projectCatalog.writeCatalog();

	currentRequest.reset();
}

auto startUpdateRequest(b3d::tools::projectServer::InternalRequest internalRequest) -> InternalRequest
{
	LOG_INFO << "Starting update request " << internalRequest.userRequest.uuid;
	auto pipelineParams = b3d::tools::sofia_nano_pipeline::SofiaNanoPipelineUpdateParams{
		.sofiaParams = internalRequest.internalParams,
		.subRegion = internalRequest.userRequest.subRegion,
		.fitsInputFilePath = internalRequest.fitsDataInputFilePath,
		.maskInputFilePath = internalRequest.fitsMaskInputFilePath,
		.inputNvdbFilePath = internalRequest.inputNvdbFilePath,
		.outputNvdbFilePath = internalRequest.outputNvdbFilePath,
		.sofiaWorkingDirectory = internalRequest.sofiaOutputDirectoryPath,
		.nanoWorkingDirectory = internalRequest.nvdbOutputDirectoryPath
	};

	internalRequest.userRequest.result =
		b3d::tools::sofia_nano_pipeline::runSearchAndUpdateNvdbSync(*sofiaProcessRunner, pipelineParams);

	LOG_INFO << "Update request " << internalRequest.userRequest.uuid << " finished "
			 << (internalRequest.userRequest.result.wasSuccess() ? "successful" : "with error");
	return internalRequest;
}

auto startCreateRequest(b3d::tools::projectServer::InternalRequest internalRequest) -> InternalRequest
{
	LOG_INFO << "Starting create request " << internalRequest.userRequest.uuid;
	auto pipelineParams = b3d::tools::sofia_nano_pipeline::SofiaNanoPipelineInitialParams{
		.sofiaParams = internalRequest.internalParams,
		.fitsInputFilePath = internalRequest.fitsDataInputFilePath,
		.outputNvdbFilePath = internalRequest.outputNvdbFilePath
	};

	internalRequest.userRequest.result =
		b3d::tools::sofia_nano_pipeline::runSearchAndCreateNvdbSync(*sofiaProcessRunner, pipelineParams);

	LOG_INFO << "Update request " << internalRequest.userRequest.uuid << " finished "
			 << (internalRequest.userRequest.result.wasSuccess() ? "successful" : "with error");
	return internalRequest;
}

auto getStatus(const httplib::Request& req, httplib::Response& res) -> void
{
	LOG_INFO << req.path << " from " << req.remote_addr;
	processCurrentRequest();

	nlohmann::json retJ;
	retJ["status"] = "OK";
	res.set_content(retJ.dump(), "application/json");
}

auto getRequestRunning(const httplib::Request& req, httplib::Response& res) -> void
{
	LOG_INFO << req.path << " from " << req.remote_addr;
	processCurrentRequest();

	nlohmann::json retJ;

	if (currentRequest == nullptr)
	{
		retJ["running_requests"] = 0;
	}
	else
	{
		retJ["running_requests"] = 1;
	}
	res.set_content(retJ.dump(), "application/json");
}

auto getProjects(const httplib::Request& req, httplib::Response& res) -> void
{
	LOG_INFO << req.path << " from " << req.remote_addr;
	processCurrentRequest();
	std::vector<b3d::tools::project::Project> projects;
	for (const auto& project : projectProvider->getProjects() | std::views::values | std::views::all)
	{
		projects.push_back(project);
	}

	LOG_INFO << "Sending Projects";
	res.set_content(nlohmann::json(projects).dump(), "application/json");
}

auto getProjectFromUuid(const httplib::Request& req, httplib::Response& res) -> void
{
	LOG_INFO << req.path << " from " << req.remote_addr;

	processCurrentRequest();
	std::lock_guard currRequestLock(currentRequestMutex);

	// ProjectUUID not provided!
	if (!req.path_params.contains("projectUUID") || req.path_params.at("projectUUID").empty())
	{
		LOG_INFO << "Missing projectUUID!";
		nlohmann::json retJ;
		retJ["message"] = "Project UUID not provided!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	const auto projectUUID = req.path_params.at("projectUUID");

	// Project does not exist!
	if (!projectProvider->hasProject(projectUUID))
	{
		LOG_INFO << "Project with UUID " << projectUUID << " not found!";

		nlohmann::json retJ;
		retJ["message"] = "Project with given projectUUID not found!";

		res.status = httplib::StatusCode::NotFound_404;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	LOG_INFO << "Sending Project " << projectUUID;

	const auto& proj = projectProvider->getProject(projectUUID);
	res.set_content(nlohmann::json(proj).dump(), "application/json");
}

auto postStartSearch([[maybe_unused]] const httplib::Request& req, httplib::Response& res, const httplib::ContentReader& contentReader)
	-> void
{
	processCurrentRequest();
	std::lock_guard currRequestLock(currentRequestMutex);

	// Ongoing request
	if (currentRequest != nullptr)
	{
		nlohmann::json retJ;
		retJ["message"] = "Ongoing request";
		res.status = httplib::StatusCode::ServiceUnavailable_503;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	std::string bodyString;
	contentReader(
		[&bodyString](const char* data, size_t data_length)
		{
			bodyString.append(data, data_length);
			return true;
		});

	auto jsonInput = nlohmann::json::parse(bodyString);

	// Input not valid
	if (jsonInput.empty() || !jsonInput.contains("projectUUID"))
	{
		nlohmann::json retJ;
		retJ["message"] = "Parameters empty or projectUUID not provided!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	std::string projectUuid = jsonInput["projectUUID"];
	if (!projectProvider->hasProject(projectUuid))
	{
		nlohmann::json retJ;
		retJ["message"] = "projectUUID not valid!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	auto& project = projectProvider->getProject(projectUuid);
	auto& catalog = projectProvider->getCatalog(project.projectUUID);

	// First successful request is original mask.
	const auto possibleRequest =
		std::ranges::find_if(project.requests, [](const auto& request) { return request.result.wasSuccess(); });
	if (possibleRequest == project.requests.end())
	{
		nlohmann::json retJ;
		retJ["message"] = "Project does not contain a valid initial result. Could not continue.";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	auto internalRequest =
		InternalRequest{ .userRequest = {},
						 .projectUUID = projectUuid,
						 .internalParams = b3d::tools::sofia::SofiaParams(b3d::tools::sofia::DEFAULT_PARAMS) };

	// Get SoFiaParams from http-request
	{
		auto err = false;
		try
		{
			internalRequest.userRequest.sofiaParameters =
				jsonInput["sofia_params"].get<b3d::tools::sofia::SofiaParams>();
		}
		catch ([[maybe_unused]] nlohmann::json::type_error& e)
		{
			err = true;
			// std::cout << e.what();
			// [json.exception.type_error.304] cannot use at() with object
		}
		catch ([[maybe_unused]] nlohmann::json::parse_error& e)
		{
			err = true;
			// std::cout << e.what();
		}
		catch ([[maybe_unused]] nlohmann::json::exception& e)
		{
			err = true;
			// std::cout << e.what();
		}
		if (err)
		{

			nlohmann::json retJ;
			retJ["message"] = "Could not parse SoFiaParams from request.";

			res.status = httplib::StatusCode::BadRequest_400;
			res.set_content(retJ.dump(), "application/json");
			return;
		}
	}

	const auto requestUuid = internalRequest.userRequest.createUUID();
	internalRequest.userRequest.uuid = requestUuid;
	internalRequest.workingDirectoryPath = project.projectPathAbsolute / "requests" / internalRequest.userRequest.uuid;
	internalRequest.sofiaOutputDirectoryPath = internalRequest.workingDirectoryPath / "sofia";


	const auto& oldRequest = *possibleRequest;

	internalRequest.fitsDataInputFilePath = catalog.getFilePathAbsolute(project.fitsOriginUUID);
	internalRequest.fitsMaskInputFilePath = catalog.getFilePathAbsolute(oldRequest.result.sofiaResult.resultFile);
	internalRequest.inputNvdbFilePath = catalog.getFilePathAbsolute(oldRequest.result.nanoResult.resultFile);
	internalRequest.nvdbOutputDirectoryPath = internalRequest.workingDirectoryPath / "nano";
	internalRequest.outputNvdbFilePath = internalRequest.nvdbOutputDirectoryPath / "out.nvdb";
	if (!std::filesystem::exists(internalRequest.workingDirectoryPath / "nano"))
	{
		std::filesystem::create_directories(internalRequest.workingDirectoryPath / "nano");
	}

	// input.region must be present. Otherwise we can't continue.
	if (!internalRequest.userRequest.sofiaParameters.containsKey("input.region"))
	{
		nlohmann::json retJ;
		retJ["message"] = "input.region not provided.";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	// Extract region from sofiaParameters
	// Calculate Offset of Subregion
	// xmin, xmax, ymin, ymax, zmin, zmax
	auto regionString = internalRequest.userRequest.sofiaParameters.getStringValue("input.region").value();

	std::array<int, 6> subRegionValues;
	// Extract subregion values from string
	{
		std::stringstream ss(regionString);
		std::string item;
		auto arrayPos = 0;
		while (std::getline(ss, item, ','))
		{
			subRegionValues[arrayPos] = std::stoi(item);
			arrayPos++;
		}
	}
	internalRequest.userRequest.subRegion =
		b3d::common::Box3I{ { subRegionValues[0], subRegionValues[2], subRegionValues[4] },
							{ subRegionValues[1], subRegionValues[3], subRegionValues[5] } };

	// Ignore the following settings
	//  - output.writeMask
	//  - output.writeRawMask
	//  - output.overwrite
	//  - output.directory
	//  - output.filename
	//  - reliability.catalog
	//  - flag.catalog
	//  - input.data -> Derived from project
	//  - input.mask -> Derived from project.request
	//  - input.gain
	//  - input.noise
	//  - input.primaryBeam
	//  - input.weights

	// Params to ignore
	const auto ignoredParams =
		std::vector<std::string>{ "output.writeMask", "output.writeRawMask", "output.overwrite", "output.directory",
								  "output.filename",  "reliability.catalog", "flag.catalog",	 "input.data",
								  "input.mask",		  "input.gain",			 "input.noise",		 "input.primaryBeam",
								  "input.weights",	  "linker.enable" };

	// Copy every parameter except the ones in ignoredParams
	for (const auto& [key, value] : internalRequest.userRequest.sofiaParameters)
	{
		if (std::ranges::find(ignoredParams, key) == ignoredParams.end())
		{
			internalRequest.internalParams.setOrReplace(key, value);
		}
		else
		{
			// Report ignored parameter
			LOG_WARNING << "Ignoring parameter " << key;
		}
	}

	if (!std::filesystem::exists(internalRequest.sofiaOutputDirectoryPath))
	{
		std::filesystem::create_directories(internalRequest.sofiaOutputDirectoryPath);
	}

	internalRequest.internalParams.setOrReplace("output.directory",
												internalRequest.sofiaOutputDirectoryPath.generic_string());
	internalRequest.internalParams.setOrReplace("output.filename", "out");
	internalRequest.internalParams.setOrReplace("output.writeMask", "true");
	// internalRequest.internalParams.setOrReplace("output.writeRawMask", "true");

	// TODO: Disables the linker. Reliability not given!
	internalRequest.internalParams.setOrReplace("linker.enable", "false");

	internalRequest.internalParams.setOrReplace("input.data", internalRequest.fitsDataInputFilePath.generic_string());
	internalRequest.internalParams.setOrReplace("input.mask", internalRequest.fitsMaskInputFilePath.generic_string());

	// Identifier already used (Same Request)
	const auto& possibleExistingRequest =
		std::ranges::find_if(project.requests, [&cm = internalRequest](const auto& request) -> bool
							 { return cm.userRequest.uuid == request.uuid; });
	if (possibleExistingRequest != project.requests.end())
	{
		auto& previousRequest = *possibleExistingRequest;
		auto forceRun = jsonInput.at("force");

		if (!previousRequest.result.wasSuccess() || (jsonInput.contains("force") && jsonInput.at("force").get<bool>()))
		{
			project.requests.erase(possibleExistingRequest);
		}
		else
		{
			LOG_INFO << "Request already in use.";
			nlohmann::json retJ;
			retJ["message"] = "requestUUID already in use.";

			res.status = httplib::StatusCode::BadRequest_400;
			res.set_content(retJ.dump(), "application/json");
			return;
		}
	}

	internalRequest.userRequest.createdAt = b3d::common::helper::getSecondsSinceEpochUtc();

	currentRequest = std::make_unique<std::future<InternalRequest>>(
		std::async(std::launch::async, &startUpdateRequest, internalRequest));

	nlohmann::json ret;
	ret["requestUUID"] = requestUuid;
	res.set_content(ret.dump(), "application/json");
}

auto getRequest(const httplib::Request& req, httplib::Response& res) -> void
{
	LOG_INFO << req.path << " from " << req.remote_addr;
	processCurrentRequest();
	std::lock_guard currRequestLock(currentRequestMutex);

	// ProjectUUID not provided!
	if (!req.path_params.contains("projectUUID") || req.path_params.at("projectUUID").empty())
	{
		LOG_INFO << "Missing projectUUID!";
		nlohmann::json retJ;
		retJ["message"] = "Project UUID not provided!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}
	const auto projectUUID = req.path_params.at("projectUUID");

	// RequestUUID not provided!
	if (!req.path_params.contains("requestUUID") || req.path_params.at("requestUUID").empty())
	{
		LOG_INFO << "Missing requestUUID!";
		nlohmann::json retJ;
		retJ["message"] = "Request UUID not provided!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}
	const auto requestUUID = req.path_params.at("requestUUID");


	// Project does not exist!
	if (!projectProvider->hasProject(projectUUID))
	{
		LOG_INFO << "Project with UUID " << projectUUID << " not found!";

		nlohmann::json retJ;
		retJ["message"] = "Project with given projectUUID not found!";

		res.status = httplib::StatusCode::NotFound_404;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	// Request with projectUUID is not finished yet!
	if (currentRequest != nullptr)
	{
		LOG_INFO << "Request not finished yet!";

		nlohmann::json retJ;
		retJ["message"] = std::format("Request not finished yet!");
		res.status = httplib::StatusCode::ServiceUnavailable_503;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	const auto& project = projectProvider->getProject(projectUUID);

	// Request does not exist!
	const auto& possibleRequest =
		std::ranges::find_if(project.requests, [&ruuid = requestUUID](const b3d::tools::project::Request& m) -> bool
							 { return ruuid == m.uuid; });
	if (possibleRequest == project.requests.end())
	{
		LOG_INFO << "Request with UUID " << requestUUID << " not found!";

		nlohmann::json retJ;
		retJ["message"] = "Request with given requestUUID not found!";
		res.status = httplib::StatusCode::NotFound_404;
		res.set_content(retJ.dump(), "application/json");
		return;
	}


	LOG_INFO << "Sending Requests " << requestUUID << "from project " << projectUUID;

	const auto request = *possibleRequest;
	nlohmann::json retJ(request);
	res.status = httplib::StatusCode::OK_200;
	res.set_content(retJ.dump(), "application/json");
}

auto getRequestsFromProjectUuid(const httplib::Request& req, httplib::Response& res) -> void
{
	LOG_INFO << req.path << " from " << req.remote_addr;
	processCurrentRequest();
	std::lock_guard currRequestLock(currentRequestMutex);

	// ProjectUUID not provided!
	if (!req.path_params.contains("projectUUID") || req.path_params.at("projectUUID").empty())
	{
		LOG_INFO << "Missing projectUUID!";
		nlohmann::json retJ;
		retJ["message"] = "Project UUID not provided!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	const auto projectUUID = req.path_params.at("projectUUID");
	// Project not found
	if (!projectProvider->hasProject(projectUUID))
	{
		LOG_INFO << "Project with UUID " << req.path_params.at("projectUUID") << " not found.";
		nlohmann::json retJ;
		retJ["message"] = "Project with given UUID not found!";

		res.status = httplib::StatusCode::NotFound_404;
		return;
	}

	LOG_INFO << "Sending Requests from project " << projectUUID;

	const auto& requests = projectProvider->getProject(projectUUID).requests;
	res.set_content(nlohmann::json(requests).dump(), "application/json");
	res.status = httplib::StatusCode::OK_200;
}

auto getFile(const httplib::Request& req, httplib::Response& res) -> void
{
	LOG_INFO << req.path << " from " << req.remote_addr;
	// ProjectUUID not provided!
	if (!req.path_params.contains("fileUUID") || req.path_params.at("fileUUID").empty())
	{
		LOG_INFO << "Missing fileUUID!";
		nlohmann::json retJ;
		retJ["message"] = "UUID for file not provided";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	auto pathToFile = std::filesystem::path{};
	for (auto& catalog : projectProvider->getAllCatalogs())
	{
		if (catalog.second.contains(req.path_params.at("fileUUID")))
		{
			pathToFile = catalog.second.getFilePathAbsolute(req.path_params.at("fileUUID"));
			break;
		}
	}
	if (pathToFile.empty())
	{
		LOG_INFO << "File with UUID " << req.path_params.at("fileUUID") << " not found.";
		nlohmann::json retJ;
		retJ["message"] = "No file found";

		res.status = httplib::StatusCode::NotFound_404;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	auto fin = new std::ifstream(pathToFile, std::ifstream::binary);
	if (!fin->good())
	{
		LOG_ERROR << "Cannot access file with UUID " << req.path_params.at("fileUUID");
		nlohmann::json retJ;
		retJ["message"] = "Could not open file.";

		res.status = httplib::StatusCode::InternalServerError_500;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	LOG_INFO << "Sending file with UUID " << req.path_params.at("fileUUID");

	const auto start = fin->tellg();
	fin->seekg(0, std::ios::end);
	auto fileSize = fin->tellg() - start;
	fin->seekg(start);

	res.set_content_provider(
		fileSize, "application/" + pathToFile.extension().string().substr(1), // Content type

		[fin, start, fileSize](size_t offset, size_t length, httplib::DataSink& sink)
		{
			if (fin->good() && !fin->eof() && offset < static_cast<size_t>(fileSize))
			{
				std::vector<char> data(std::min((size_t)10240, length));

				fin->read(data.data(), std::min(data.size(), fileSize - offset));
				auto l = fin->tellg();
				sink.write(data.data(), std::min(data.size(), fileSize - offset));
			}
			else
			{
				sink.done(); // No more data
			}
			return true; // return 'false' if you want to cancel the process.
		},
		[fin]([[maybe_unused]] bool success)
		{
			fin->close();
			delete fin;
		});
}

auto postNewProject([[maybe_unused]] const httplib::Request& req,[[maybe_unused]]  httplib::Response& res, const httplib::ContentReader& contentReader)
	-> void
{
	// Create Random project UUID
	std::random_device rd;
	auto seed_data = std::array<int, std::mt19937::state_size>{};
	std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
	std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
	std::mt19937 generator(seq);
	uuids::uuid_random_generator gen{ generator };

	std::string projectUUIDString = uuids::to_string(gen());

	b3d::tools::project::Project newProject;
	newProject.projectUUID = projectUUIDString;
	newProject.projectPathAbsolute = projectProvider->getProjectsPathAbsolute() / newProject.projectUUID;

	b3d::tools::project::catalog::FileCatalog newCatalog =
		b3d::tools::project::catalog::FileCatalog::createOrLoadCatalogInDirectory(newProject.projectPathAbsolute);

	newProject.projectName = "NEW";
	newProject.fitsOriginFileName = "original.fits";
	newProject.fitsOriginUUID =
		newCatalog.addFilePathAbsolute(newProject.projectPathAbsolute / newProject.fitsOriginFileName);

	std::filesystem::create_directories(newProject.projectPathAbsolute / "requests");

	newCatalog.writeCatalog();
	{
		const auto projectFilePath = newProject.projectPathAbsolute / "project.json";
		std::ofstream ofs(projectFilePath, std::ofstream::trunc);
		nlohmann::json j = newProject;
		ofs << std::setw(4) << j << std::endl;
		ofs.close();
	}

	projectProvider->addExistingProject(newProject.projectUUID);
	// Create new Project
	// create new catalog
	// Create new Directory for project
	// Add Project to projectprovider
	auto& createdProject = projectProvider->getProject(newProject.projectUUID);
	createdProject.projectPathAbsolute = newProject.projectPathAbsolute;
	auto fout = new std::ofstream(createdProject.projectPathAbsolute / createdProject.fitsOriginFileName,
								  std::ifstream::binary);

	contentReader(
		[&](const char* data, size_t data_length)
		{
			fout->write(data, data_length);
			return true;
		});
	fout->close();

	createdProject.fitsOriginProperties =
		b3d::tools::fits::getFitsProperties(createdProject.projectPathAbsolute / createdProject.fitsOriginFileName);

	auto request = b3d::tools::project::Request{ .uuid = uuids::to_string(gen()),
												 .subRegion = { createdProject.fitsOriginProperties.axisDimensions[0],
																createdProject.fitsOriginProperties.axisDimensions[1],
																createdProject.fitsOriginProperties.axisDimensions[2] },
												 .sofiaParameters = {},
												 .result = {},
												 .createdAt = b3d::common::helper::getSecondsSinceEpochUtc() };


	std::filesystem::create_directories(createdProject.projectPathAbsolute / "requests" / request.uuid / "nano");
	request.result.nanoResult = b3d::tools::nano::convertFitsToNano(
		createdProject.projectPathAbsolute / createdProject.fitsOriginFileName,
		createdProject.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb");
	if (!request.result.nanoResult.wasSuccess())
	{
		LOG_ERROR << "Failed to create NVDB.";
		return;
	}
	request.result.nanoResult.resultFile = newCatalog.addFilePathAbsolute(
		createdProject.projectPathAbsolute / "requests" / request.uuid / "nano" / "out.nvdb");
	createdProject.requests.push_back(request);
	newCatalog.writeCatalog();
	projectProvider->saveProject(createdProject.projectUUID);
}

auto deleteProject(const httplib::Request& req, httplib::Response& res) -> void
{
	// ProjectUUID not provided!
	if (!req.path_params.contains("projectUUID") || req.path_params.at("projectUUID").empty())
	{
		LOG_INFO << "Missing projectUUID!";
		nlohmann::json retJ;
		retJ["message"] = "Project UUID not provided!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}
	const auto projectUUID = req.path_params.at("projectUUID");

	// Project does not exist!
	if (!projectProvider->hasProject(projectUUID))
	{
		LOG_INFO << "Project with UUID " << projectUUID << " not found!";

		nlohmann::json retJ;
		retJ["message"] = "Project with given projectUUID not found!";

		res.status = httplib::StatusCode::NotFound_404;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	projectProvider->removeProject(projectUUID);
	res.status = httplib::StatusCode::OK_200;
}

auto putChangeProject(const httplib::Request& req, httplib::Response& res, const httplib::ContentReader& contentReader)
	-> void
{
	// ProjectUUID not provided!
	if (!req.path_params.contains("projectUUID") || req.path_params.at("projectUUID").empty())
	{
		LOG_INFO << "Missing projectUUID!";
		nlohmann::json retJ;
		retJ["message"] = "Project UUID not provided!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}
	const auto projectUUID = req.path_params.at("projectUUID");

	// Project does not exist!
	if (!projectProvider->hasProject(projectUUID))
	{
		LOG_INFO << "Project with UUID " << projectUUID << " not found!";

		nlohmann::json retJ;
		retJ["message"] = "Project with given projectUUID not found!";

		res.status = httplib::StatusCode::NotFound_404;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	std::string bodyString;
	contentReader(
		[&bodyString](const char* data, size_t data_length)
		{
			bodyString.append(data, data_length);
			return true;
		});

	auto jsonInput = nlohmann::json::parse(bodyString);

	// Input not valid
	if (jsonInput.empty() || !jsonInput.contains("projectName"))
	{
		nlohmann::json retJ;
		retJ["message"] = "Parameters empty or projectName not provided!";

		res.status = httplib::StatusCode::BadRequest_400;
		res.set_content(retJ.dump(), "application/json");
		return;
	}

	auto& proj = projectProvider->getProject(projectUUID);
	proj.projectName = jsonInput["projectName"];
	[[maybe_unused]] auto saved = projectProvider->saveProject(projectUUID);
	res.status = httplib::StatusCode::OK_200;
}


auto main(const int argc, char** argv) -> int
{
	static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
	static plog::ColorConsoleAppender<plog::TxtFormatter> colorConsoleAppender;

	plog::init(plog::debug, &colorConsoleAppender);

	LOG_NONE << "Starting ProjectServer!";

	args::ArgumentParser parser("SoFiA-2 Wrapper Server", "");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::ValueFlag<std::string> sofiaPathArgument(parser, "path/to/sofia/executable",
												   "Absolute path to sofia executable", { "sofia-executable", "se" });
	args::ValueFlag<std::string> commonRootPathArgument(
		parser, "common/root/path",
		"Common root path where shared data is located and written to. All relative paths starting from here.",
		{ "root-path", "rp" });
	args::ValueFlag<int> serverListeningPortArgument(parser, "5051", "Port the server is listening", { "port", "p" },
													 5051);

	// Argument parsing & validation
	{
		try
		{
			parser.ParseCLI(argc, argv);
		}
		catch (args::Help)
		{
			LOG_NONE << parser;
			return EXIT_SUCCESS;
		}
		catch (args::ParseError e)
		{
			LOG_FATAL << e.what();
			LOG_FATAL << parser;
			return EXIT_FAILURE;
		}
		catch (args::ValidationError e)
		{
			LOG_FATAL << e.what();
			LOG_FATAL << parser;
			return EXIT_FAILURE;
		}

		if (sofiaPathArgument)
		{
			sofiaPath = args::get(sofiaPathArgument);
		}

		if (commonRootPathArgument)
		{
			commonRootPath = args::get(commonRootPathArgument);
		}

		if (sofiaPath.empty() || !std::filesystem::exists(sofiaPath))
		{
			LOG_FATAL << "No path to SoFiA - 2 executable !";
			LOG_FATAL << parser;
			return EXIT_FAILURE;
		}

		LOG_NONE << "Using " << sofiaPath << " as SoFiA executable";

		if (commonRootPath.empty())
		{
			LOG_FATAL << "No common root path!";
			LOG_FATAL << parser;
			return EXIT_FAILURE;
		}

		LOG_NONE << "Using " << commonRootPath << " as common root path";
	}

	projectProvider = std::make_unique<ProjectProvider>(std::filesystem::path{ args::get(commonRootPathArgument) });
	sofiaProcessRunner = std::make_unique<b3d::tools::sofia::SofiaProcessRunner>(sofiaPath);

	httplib::Server svr;

	// Error
	svr.set_exception_handler(
		[]([[maybe_unused]] const auto& req, auto& res, std::exception_ptr ep)
		{
			const auto fmt = "<h1>Error 500</h1><p>{}</p>";
			std::string message = "";
			try
			{

				std::rethrow_exception(ep);
			}
			catch (std::exception& e)
			{
				LOG_ERROR << e.what();
				message = std::format(fmt, e.what());
			}
			catch (...)
			{
				LOG_ERROR << "Unknown Exception";
				message = std::format(fmt, "Unknown Exception");
			}

			res.set_content(message, "text/html");
			res.status = httplib::StatusCode::InternalServerError_500;
		});

	svr.Get("/status", &getStatus);
	svr.Get("/requestRunning", &getRequestRunning);
	// svr.Get("/catalog", &getCatalog);
	svr.Get("/projects", &getProjects);
	svr.Get("/project/:projectUUID", &getProjectFromUuid);
	svr.Post("/startSearch", &postStartSearch);
	svr.Get("/request/:projectUUID/:requestUUID", &getRequest);
	svr.Get("/requests/:projectUUID", &getRequestsFromProjectUuid);
	svr.Get("/file/:fileUUID", &getFile);
	svr.Post("/project/new", &postNewProject);
	svr.Delete("/project/:projectUUID", &deleteProject);
	svr.Put("/project/:projectUUID", &putChangeProject);

	LOG_NONE << "Server is listening on port " << args::get(serverListeningPortArgument);
	svr.listen("0.0.0.0", args::get(serverListeningPortArgument));
}
