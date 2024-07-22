#include <string>
#include <vector>

#include <httplib.h>
#include <map>

#include <nlohmann/json.hpp>

#include <filesystem>

#include <future>


#include <algorithm>

#include <args.hxx>

#include "Project.h"
#include "ProjectProvider.h"

#include "SoFiA.h"

// https://stackoverflow.com/questions/2989810/which-cross-platform-preprocessor-defines-win32-or-win32-or-win32
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix))
std::filesystem::path sofiaPath = boost::process::search_path("sofia").string();
#else
std::filesystem::path sofiaPath = boost::process::search_path("sofia.exe").string();

#endif


auto commonRootPath = boost::process::filesystem::path("");
std::unique_ptr<std::future<b3d::tools::sofiasearch::ProcessResult>> currentRequest{nullptr};
std::unique_ptr<b3d::tools::projectexplorer::ProjectProvider> projectProvider{ nullptr };

std::mutex currentRequestMutex;

auto processCurrentRequest() -> void
{
	std::lock_guard lock(currentRequestMutex);
	using namespace std::chrono_literals;
	if (!currentRequest)
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
	
	auto req = currentRequest->get();
	projectProvider->getProject(req.projectIdentifier).requests.emplace_back(req.request);
	projectProvider->saveProject(req.projectIdentifier);
	projectProvider->saveRootCatalog();	
	
	currentRequest.reset();
}

auto main(const int argc, char** argv) -> int
{

	args::ArgumentParser parser("SoFiA-2 Wrapper Server", "");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	args::ValueFlag<std::string> sofiaPathArgument(parser, "path/to/sofia/executable", "Absolute path to sofia executable", { "sofia-executable","se" });
	args::ValueFlag<std::string> commonRootPathArgument(parser, "common/root/path", "Common root path where shared data is located and written to. All relative paths starting from here.", { "root-path", "rp" });
	args::ValueFlag<int> serverListeningPortArgument(parser, "5051","Port the server is listening", { "port", "p" }, 5051);

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

	if (sofiaPath.empty())
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

	std::filesystem::path root{ args::get(commonRootPathArgument) };
	projectProvider = std::make_unique<b3d::tools::projectexplorer::ProjectProvider>(root);
	
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

	svr.Get("/catalog",
			[](const httplib::Request& req, httplib::Response& res)
			{ res.set_content(nlohmann::json(projectProvider->getRootCatalog()).dump(), "application/json");
			});

	svr.Get("/projects",
			[](const httplib::Request& req, httplib::Response& res)
			{
				std::vector<b3d::tools::projectexplorer::Project> projects;
				for (const auto& project : projectProvider->getProjects() | std::views::values | std::views::all)
				{
					projects.push_back(project);
				}
				res.set_content(nlohmann::json(projects).dump(), "application/json");
			});

	svr.Get("/project/:guid",
			[](const httplib::Request& req, httplib::Response& res)
	{
				processCurrentRequest();
				std::lock_guard currRequestLock(currentRequestMutex);

				if (!req.path_params.contains("guid"))
				{
					res.status = httplib::StatusCode::BadRequest_400;
					return;
				}
				const auto guidVal = req.path_params.at("guid");
				if (!projectProvider->hasProject(guidVal))
				{
					res.status = httplib::StatusCode::NotFound_404;
					return;
				}
				const auto& proj = projectProvider->getProject(guidVal);
				res.set_content(nlohmann::json(proj).dump(), "application/json");
	});

	// 
	svr.Post("/start",
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
				if (jsonInput.empty() || !jsonInput.contains("projectGUID"))
				{
					nlohmann::json retJ;
					retJ["message"] = "Parameters empty or projectGUID not provided!";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				std::string projectGuid = jsonInput["projectGUID"];
				if (!projectProvider->hasProject(projectGuid))
				{
					nlohmann::json retJ;
					retJ["message"] = "projectGUID not valid!";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				auto& project = projectProvider->getProject(projectGuid);
				auto& cat = projectProvider->getRootCatalog();


				const auto& filePath = cat.getFilePathAbsolute(project.fitsOriginGUID);

				// Build new search
				b3d::tools::projectexplorer::Request sofiaRequest;
				sofiaRequest.preSofiaSearchParameters.emplace_back("../../sofia.par");
				sofiaRequest.preSofiaSearchParameters.emplace_back(std::format("input.data={}", filePath.string()));
				sofiaRequest.preSofiaSearchParameters.emplace_back(std::format("output.filename={}", "out"));


				for (auto& [key, value] : jsonInput["sofia_params"].items())
				{
					b3d::tools::sofiasearch::appendParameterToSoFiARequest(sofiaRequest, key, value.get<std::string>());
				}

				const auto sofiaRequestIdentifier = sofiaRequest.createUUID();
				
				sofiaRequest.guid = sofiaRequestIdentifier;
				sofiaRequest.sofiaExecutablePath = sofiaPath;
				sofiaRequest.workingDirectory = project.projectPathAbsolute / "requests" / sofiaRequestIdentifier;
				sofiaRequest.preSofiaSearchParameters.emplace_back(
					std::format("output.directory={}", sofiaRequest.workingDirectory.string()));

				// Identifier already used (Same Request)
				const auto& possibleRequest =
					std::ranges::find_if(project.requests,
										 [&cm = sofiaRequest](const b3d::tools::projectexplorer::Request& m) -> bool
										 { return cm.guid == m.guid; });
				if (possibleRequest != project.requests.end())
				{
					auto& previousRequest = *possibleRequest;
					if (previousRequest.result.returnCode != 0)
					{
						project.requests.erase(possibleRequest);
					}
					else
					{
						nlohmann::json retJ;
						retJ["message"] = "requestGUID already in use.";

						res.status = httplib::StatusCode::BadRequest_400;
						res.set_content(retJ.dump(), "application/json");
						return;	
					}
				}
								
				currentRequest = std::make_unique<std::future<b3d::tools::sofiasearch::ProcessResult>>(
					std::async(std::launch::async, b3d::tools::sofiasearch::RequestProcessor(), std::ref(project),
							   std::ref(cat), sofiaRequestIdentifier, std::move(sofiaRequest)));

				nlohmann::json ret;
				ret["requestGUID"] = sofiaRequestIdentifier;
				res.set_content(ret.dump(), "application/json");

			 });


	svr.Get("/result/:projectGUID/:requestGUID",
			 [](const httplib::Request& req, httplib::Response& res)
			 {
				 processCurrentRequest();
				 std::lock_guard currRequestLock(currentRequestMutex);

				 // ProjectGUID not provided!
				 if (!req.path_params.contains("projectGUID"))
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "Project GUID not provided!";

					 res.status = httplib::StatusCode::BadRequest_400;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }
				 const auto projectGUID = req.path_params.at("projectGUID");

			 	 // RequestGUID not provided!
				 if (!req.path_params.contains("requestGUID"))
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "Request GUID not provided!";

					 res.status = httplib::StatusCode::BadRequest_400;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }
				 const auto requestGUID = req.path_params.at("requestGUID");


				 // Project does not exist!
				 if (!projectProvider->hasProject(projectGUID))
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "Project with given projectGUID not found!";

					 res.status = httplib::StatusCode::NotFound_404;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }
				 			 
				 
				 // Request with projectGUID is not finished yet!
				 if (currentRequest != nullptr)
				 {
					 nlohmann::json retJ;
					 retJ["message"] = std::format("Request not finished yet!");
					 res.status = httplib::StatusCode::ServiceUnavailable_503;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }
				 
				 
				 const auto& project = projectProvider->getProject(projectGUID);

				 // Request does not exist!
				 const auto& possibleRequest =
					 std::ranges::find_if(project.requests,
										  [&rguid = requestGUID](const b3d::tools::projectexplorer::Request& m) -> bool
										  { return rguid == m.guid; });
				 if (possibleRequest == project.requests.end())
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "Request with given requestGUID not found!";
					 res.status = httplib::StatusCode::NotFound_404;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }

				 const auto request = *possibleRequest;
				 nlohmann::json retJ(request);
				 res.status = httplib::StatusCode::OK_200;
				 res.set_content(retJ.dump(), "application/json");

			 });

	svr.Get("/results/:projectGUID", [](const httplib::Request& req, httplib::Response& res)
			{
				processCurrentRequest();
				std::lock_guard currRequestLock(currentRequestMutex);

				// ProjectGUID not provided!
				if (!req.path_params.contains("projectGUID"))
				{
					nlohmann::json retJ;
					retJ["message"] = "Project GUID not provided!";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				const auto projectGUID = req.path_params.at("projectGUID");
				// Project not found
				if (!projectProvider->hasProject(projectGUID))
				{
					nlohmann::json retJ;
					retJ["message"] = "Project with given GUID not found!";

					res.status = httplib::StatusCode::NotFound_404;
					return;
				}

				const auto& requests = projectProvider->getProject(projectGUID).requests;
				res.set_content(nlohmann::json(requests).dump(), "application/json");
				res.status = httplib::StatusCode::OK_200;

			});

	svr.Get("/file/:fileGUID",
			[](const httplib::Request& req, httplib::Response& res)
			{
				// ProjectGUID not provided!
				if (!req.path_params.contains("fileGUID"))
				{
					nlohmann::json retJ;
					retJ["message"] = "GUID for file not provided";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}


				const auto pathToFile = projectProvider->getRootCatalog().getFilePathAbsolute(req.path_params.at("fileGUID"));
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

				res.set_content_provider(fileSize, "application/nvdb", // Content type
											
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
