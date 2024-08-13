#include <filesystem>
#include <future>
#include <string>
#include <vector>

#include <args.hxx>
#include <httplib.h>
#include <boost/process.hpp>

#include "SofiaNanoPipeline.h"
#include "SofiaProcessRunner.h"

#include "Internals.h"
#include "ProjectProvider.h"

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
		return;
	}

	// Save request and modify paths to UUIDs

	auto req = currentRequest->get();

	if (req.userRequest.result.sofiaResult.wasSuccess())
	{
		req.userRequest.result.sofiaResult.resultFile = projectProvider->getCatalog().addFilePathAbsolute(
			req.userRequest.result.sofiaResult.resultFile);
	}

	if (req.userRequest.result.nanoResult.wasSuccess())
	{
		if(req.userRequest.result.sofiaResult.returnCode == 8)
		{
			req.userRequest.result.sofiaResult.resultFile =
				projectProvider->getCatalog().addFilePathAbsolute(req.userRequest.result.sofiaResult.resultFile);
		}

		req.userRequest.result.nanoResult.resultFile =
			projectProvider->getCatalog().addFilePathAbsolute(req.userRequest.result.nanoResult.resultFile);
	}

	projectProvider->getProject(req.projectUUID).requests.emplace_back(req.userRequest);
	projectProvider->saveProject(req.projectUUID);
	projectProvider->saveRootCatalog();
	
	currentRequest.reset();
}


auto startUpdateRequest(b3d::tools::projectServer::InternalRequest internalRequest)
	-> InternalRequest
{
	auto pipelineParams = b3d::tools::sofia_nano_pipeline::SofiaNanoPipelineUpdateParams{
		.sofiaParams = internalRequest.internalParams,
		.fitsInputFilePath = internalRequest.fitsDataInputFilePath,
		.maskInputFilePath = internalRequest.fitsMaskInputFilePath,
		.inputNvdbFilePath = internalRequest.inputNvdbFilePath,
		.outputNvdbFilePath = internalRequest.outputNvdbFilePath
	};

	internalRequest.userRequest.result = b3d::tools::sofia_nano_pipeline::runSearchAndUpdateNvdbSync(*sofiaProcessRunner, pipelineParams);

	return internalRequest;
}

auto main(const int argc, char** argv) -> int
{
	args::ArgumentParser parser("SoFiA-2 Wrapper Server", "");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::ValueFlag<std::string> sofiaPathArgument(parser, "path/to/sofia/executable", "Absolute path to sofia executable", { "sofia-executable","se" });
	args::ValueFlag<std::string> commonRootPathArgument(parser, "common/root/path", "Common root path where shared data is located and written to. All relative paths starting from here.", { "root-path", "rp" });
	args::ValueFlag<int> serverListeningPortArgument(parser, "5051","Port the server is listening", { "port", "p" }, 5051);

	// Argument parsing & validation
	{
		try
		{
			parser.ParseCLI(argc, argv);
		}
		catch (args::Help)
		{
			std::cout << parser;
			return EXIT_SUCCESS;
		}
		catch (args::ParseError e)
		{
			std::cerr << e.what() << std::endl;
			std::cerr << parser;
			return EXIT_FAILURE;
		}
		catch (args::ValidationError e)
		{
			std::cerr << e.what() << std::endl;
			std::cerr << parser;
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
			std::cerr << "No path to SoFiA-2 executable!\n";
			std::cout << parser;
			return EXIT_FAILURE;
		}
		std::cout << "Using " << sofiaPath << " as SoFiA executable\n";

		if (commonRootPath.empty())
		{
			std::cerr << "No common root path!\n";
			std::cout << parser;
			return EXIT_FAILURE;
		}

		std::cout << "Using " << commonRootPath << " as common root path\n";
	}

	std::filesystem::path root{ args::get(commonRootPathArgument) };

	projectProvider = std::make_unique<ProjectProvider>(root);
	sofiaProcessRunner = std::make_unique<b3d::tools::sofia::SofiaProcessRunner>(sofiaPath);
	
	httplib::Server svr;
		
	// Error
	svr.set_exception_handler(
		[](const auto& req, auto& res, std::exception_ptr ep)
		{
			auto fmt = "<h1>Error 500</h1><p>%s</p>";
			char buf[BUFSIZ];
			try
			{
				std::rethrow_exception(ep);
			}
			catch (std::exception& e)
			{
				snprintf(buf, sizeof(buf), fmt, e.what());
			}
			catch (...)
			{ // See the following NOTE
				snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
			}
			res.set_content(buf, "text/html");
			res.status = httplib::StatusCode::InternalServerError_500;
		});

	svr.Get("/status",
			[](const httplib::Request& req, httplib::Response& res)
			{
				nlohmann::json retJ;
				retJ["status"] = "OK";
				res.set_content(retJ.dump(), "application/json");
			});

	svr.Get("/catalog",
			[](const httplib::Request& req, httplib::Response& res)
			{
				res.set_content(nlohmann::json(projectProvider->getCatalog()).dump(), "application/json");
			});

	svr.Get("/projects",
			[](const httplib::Request& req, httplib::Response& res)
			{
				std::vector<b3d::tools::project::Project> projects;
				for (const auto& project : projectProvider->getProjects() | std::views::values | std::views::all)
				{
					projects.push_back(project);
				}
					res.set_content(nlohmann::json(projects).dump(), "application/json");
			});

	svr.Get("/project/:uuid",
			[](const httplib::Request& req, httplib::Response& res)
	{
				processCurrentRequest();
				std::lock_guard currRequestLock(currentRequestMutex);

				if (!req.path_params.contains("uuid"))
				{
					res.status = httplib::StatusCode::BadRequest_400;
					return;
				}
				const auto uuidVal = req.path_params.at("uuid");
				if (!projectProvider->hasProject(uuidVal))
				{
					res.status = httplib::StatusCode::NotFound_404;
					return;
				}
				const auto& proj = projectProvider->getProject(uuidVal);
					res.set_content(nlohmann::json(proj).dump(), "application/json");
	});

	// 
	svr.Post("/startSearch",
			 [](const httplib::Request& req, httplib::Response& res,
							  const httplib::ContentReader& content_reader)
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
				 content_reader(
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
				auto& catalog = projectProvider->getCatalog();


			 	// First successful request is original mask.
				const auto possibleRequest = std::ranges::find_if(project.requests, [](const auto& request) { return request.result.wasSuccess(); });
				if (possibleRequest == project.requests.end())
				{
					nlohmann::json retJ;
					retJ["message"] = "Project does not contain a valid initial result. Could not continue.";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}
				
				auto internalRequest = InternalRequest{
					.userRequest = {},
					.projectUUID = projectUuid,
					.internalParams = b3d::tools::sofia::SofiaParams(b3d::tools::sofia::DEFAULT_PARAMS)
				};

				// Get SoFiaParams from http-request
				{
					auto err = false;
					try
					{
						internalRequest.userRequest.sofiaParameters =
							jsonInput["sofia_params"].get<b3d::tools::sofia::SofiaParams>();
					}
					catch (nlohmann::json::type_error& e)
					{
						err = true; 
						// std::cout << e.what();
						// [json.exception.type_error.304] cannot use at() with object
					}
					catch (nlohmann::json::parse_error& e)
					{
						err = true;
						// std::cout << e.what();
					}
					catch (nlohmann::json::exception& e)
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
				internalRequest.outputNvdbFilePath = internalRequest.sofiaOutputDirectoryPath / "out.nvdb";

				
			 	// input.region must be present. Otherwise we can't continue.
				if (!internalRequest.userRequest.sofiaParameters.containsKey("input.region"))
				{
					nlohmann::json retJ;
					retJ["message"] = "input.region not provided.";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}
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
				const auto ignoredParams = std::vector<std::string>{
					"output.writeMask", "output.writeRawMask", "output.overwrite", "output.directory", "output.filename",
					"reliability.catalog", "flag.catalog", "input.data", "input.mask", "input.gain", "input.noise",
					"input.primaryBeam", "input.weights", "linker.enable"
				};

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


				internalRequest.internalParams.setOrReplace("input.data",
															internalRequest.fitsDataInputFilePath.generic_string());
				internalRequest.internalParams.setOrReplace("input.mask",
															internalRequest.fitsMaskInputFilePath.generic_string());

				// Identifier already used (Same Request)
				const auto& possibleExistingRequest = std::ranges::find_if(project.requests,
																   [&cm = internalRequest](const auto& request) -> bool
																   { return cm.userRequest.uuid == request.uuid; });
				if (possibleExistingRequest != project.requests.end())
				{
					auto& previousRequest = *possibleExistingRequest;
					if (previousRequest.result.returnCode != 0) // TODO: Only if forced!?
					{
						project.requests.erase(possibleExistingRequest);
					}
					else
					{
						nlohmann::json retJ;
						retJ["message"] = "requestUUID already in use.";

						res.status = httplib::StatusCode::BadRequest_400;
						res.set_content(retJ.dump(), "application/json");
						return;	
					}
				}

				currentRequest = std::make_unique<std::future<InternalRequest>>(
									 std::async(std::launch::async, &startUpdateRequest, internalRequest));

				nlohmann::json ret;
				ret["requestUUID"] = requestUuid;
				res.set_content(ret.dump(), "application/json");

			 });

	svr.Get("/request/:projectUUID/:requestUUID",
			 [](const httplib::Request& req, httplib::Response& res)
			 {
				 processCurrentRequest();
				 std::lock_guard currRequestLock(currentRequestMutex);

				 // ProjectUUID not provided!
				 if (!req.path_params.contains("projectUUID"))
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "Project UUID not provided!";

					 res.status = httplib::StatusCode::BadRequest_400;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }
				 const auto projectUUID = req.path_params.at("projectUUID");

			 	 // RequestUUID not provided!
				 if (!req.path_params.contains("requestUUID"))
				 {
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
					 nlohmann::json retJ;
					 retJ["message"] = "Project with given projectUUID not found!";

					 res.status = httplib::StatusCode::NotFound_404;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }
				 			 
				 
				 // Request with projectUUID is not finished yet!
				 if (currentRequest != nullptr)
				 {
					 nlohmann::json retJ;
					 retJ["message"] = std::format("Request not finished yet!");
					 res.status = httplib::StatusCode::ServiceUnavailable_503;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }
				 
				 
				 const auto& project = projectProvider->getProject(projectUUID);

				 // Request does not exist!
				 const auto& possibleRequest =
					 std::ranges::find_if(project.requests,
										  [&ruuid = requestUUID](const b3d::tools::project::Request& m) -> bool
										  { return ruuid == m.uuid; });
				 if (possibleRequest == project.requests.end())
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "Request with given requestUUID not found!";
					 res.status = httplib::StatusCode::NotFound_404;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }

				 const auto request = *possibleRequest;
				 nlohmann::json retJ(request);
				 res.status = httplib::StatusCode::OK_200;
				 res.set_content(retJ.dump(), "application/json");

			 });

	svr.Get("/requests/:projectUUID", [](const httplib::Request& req, httplib::Response& res)
	{
		processCurrentRequest();
		std::lock_guard currRequestLock(currentRequestMutex);

		// ProjectUUID not provided!
		if (!req.path_params.contains("projectUUID"))
		{
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
			nlohmann::json retJ;
			retJ["message"] = "Project with given UUID not found!";

			res.status = httplib::StatusCode::NotFound_404;
			return;
		}

		const auto& requests = projectProvider->getProject(projectUUID).requests;
		res.set_content(nlohmann::json(requests).dump(), "application/json");
		res.status = httplib::StatusCode::OK_200;
	});

	svr.Get("/file/:fileUUID",
			[](const httplib::Request& req, httplib::Response& res)
			{
				// ProjectUUID not provided!
				if (!req.path_params.contains("fileUUID"))
				{
					nlohmann::json retJ;
					retJ["message"] = "UUID for file not provided";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				const auto pathToFile = projectProvider->getCatalog().getFilePathAbsolute(req.path_params.at("fileUUID"));
				if (pathToFile.empty())
				{
					nlohmann::json retJ;
					retJ["message"] = "No file found";

					res.status = httplib::StatusCode::NotFound_404;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				auto fin = new std::ifstream(pathToFile, std::ifstream::binary);
				if (!fin->good())
				{
					nlohmann::json retJ;
					retJ["message"] = "Could not open file.";

					res.status = httplib::StatusCode::InternalServerError_500;
					res.set_content(retJ.dump(), "application/json");
					return;
				}
				const auto start = fin->tellg();
				fin->seekg(0, std::ios::end);
				auto fileSize = fin->tellg() - start;
				fin->seekg(start);
				;
				res.set_content_provider(
					fileSize, "application/" + pathToFile.extension().string().substr(1), // Content type
											
				 [fin, start, fileSize](size_t offset, size_t length, httplib::DataSink& sink)
				 {
				 	if (fin->good() && !fin->eof() && offset < fileSize)
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
					[fin](bool success)
					{
						fin->close();
						delete fin;
					});
			});

	std::cout << "Server is listening on port " << args::get(serverListeningPortArgument) << "\n";
	svr.listen("0.0.0.0", args::get(serverListeningPortArgument));
}
