#include <string>
#include <vector>

#include <httplib.h>
#include <map>

#include <nlohmann/json.hpp>

#include <filesystem>

#include <future>
#include <algorithm>
#include <args.hxx>

#include <boost/process.hpp>

namespace bp = boost::process;

// https://stackoverflow.com/questions/2989810/which-cross-platform-preprocessor-defines-win32-or-win32-or-win32
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix))
	auto sofiaPath = boost::process::search_path("sofia");
#else
	auto sofiaPath = boost::process::search_path("sofia.exe");
#endif

auto commonRootPath = boost::process::filesystem::path("");

const std::array<std::string, 9> sofia_return_code_messages = {
	"The pipeline successfully completed without any error.",
	"An unclassified failure occurred.",
	"A NULL pointer was encountered.",
	"A memory allocation error occurred. This could indicate that the data cube is too large for the amount of memory available on the machine.",
	"An array index was found to be out of range.",
	"An error occurred while trying to read or write a file or check if a directory or file is accessible.",
	"The overflow of an integer value occurred.",
	"The pipeline had to be aborted due to invalid user input. This could, e.g., be due to an invalid parameter setting or the wrong input file being provided.",
	"No specific error occurred, but sources were not detected either."
};

const std::array<std::string, 102> sofia_parameter_keys = { "pipeline.verbose",
														"pipeline.pedantic",
														"pipeline.threads",
														"input.data",
														"input.region",
														"input.gain",
														"input.noise",
														"input.weights",
														"input.primaryBeam",
														"input.mask",
														"input.invert",
														"flag.region",
														"flag.catalog",
														"flag.radius",
														"flag.auto",
														"flag.threshold",
														"flag.log",
														"contsub.enable",
														"contsub.order",
														"contsub.threshold",
														"contsub.shift",
														"contsub.padding",
														"scaleNoise.enable",
														"scaleNoise.mode",
														"scaleNoise.statistic",
														"scaleNoise.fluxRange",
														"scaleNoise.windowXY",
														"scaleNoise.windowZ",
														"scaleNoise.gridXY",
														"scaleNoise.gridZ",
														"scaleNoise.interpolate",
														"scaleNoise.scfind",
														"rippleFilter.enable",
														"rippleFilter.statistic",
														"rippleFilter.windowXY",
														"rippleFilter.windowZ",
														"rippleFilter.gridXY",
														"rippleFilter.gridZ",
														"rippleFilter.interpolate",
														"scfind.enable",
														"scfind.kernelsXY",
														"scfind.kernelsZ",
														"scfind.threshold",
														"scfind.replacement",
														"scfind.statistic",
														"scfind.fluxRange",
														"threshold.enable",
														"threshold.threshold",
														"threshold.mode",
														"threshold.statistic",
														"threshold.fluxRange",
														"linker.enable",
														"linker.radiusXY",
														"linker.radiusZ",
														"linker.minSizeXY",
														"linker.minSizeZ",
														"linker.maxSizeXY",
														"linker.maxSizeZ",
														"linker.minPixels",
														"linker.maxPixels",
														"linker.minFill",
														"linker.maxFill",
														"linker.positivity",
														"linker.keepNegative",
														"reliability.enable",
														"reliability.parameters",
														"reliability.threshold",
														"reliability.scaleKernel",
														"reliability.minSNR",
														"reliability.minPixels",
														"reliability.autoKernel",
														"reliability.iterations",
														"reliability.tolerance",
														"reliability.catalog",
														"reliability.plot",
														"reliability.debug",
														"dilation.enable",
														"dilation.iterationsXY",
														"dilation.iterationsZ",
														"dilation.threshold",
														"parameter.enable",
														"parameter.wcs",
														"parameter.physical",
														"parameter.prefix",
														"parameter.offset",
														"output.directory",
														"output.filename",
														"output.writeCatASCII",
														"output.writeCatXML",
														"output.writeCatSQL",
														"output.writeNoise",
														"output.writeFiltered",
														"output.writeMask",
														"output.writeMask2d",
														"output.writeRawMask",
														"output.writeMoments",
														"output.writeCubelets",
														"output.writePV",
														"output.writeKarma",
														"output.marginCubelets",
														"output.thresholdMom12",
														"output.overwrite" };


const std::array<std::string, 9> sofia_path_parameter_keys = {
	"input.data",
	"input.gain",
	"input.mask",
	"input.noise",
	"input.primaryBeam",
	"input.weights",
	"flag.catalog",
	"reliability.catalog",
	"output.directory",
};


struct SofiaResult
{
	bool finished { false };
	int returnCode { -1 };
	

	auto wasSuccess() const -> bool
	{
		return finished && returnCode == 0;
	}

	auto message() const ->std::string_view
	{
		if (0 <= returnCode && returnCode < sofia_return_code_messages.size())
		{
			return sofia_return_code_messages[1];
		}
		return sofia_return_code_messages[returnCode];
	}
};

void to_json(nlohmann::json& j, const SofiaResult& result)
{
	j = nlohmann::json
	{
		{ "finished", result.finished },
		{"returnCode", result.returnCode },
		{ "message", result.message() }
	};
}

void from_json(const nlohmann::json& j, SofiaResult& result)
{
	j.at("finished").get_to(result.finished);
	j.at("returnCode").get_to(result.returnCode);
}

struct SofiaSearch
{
	std::vector<std::string> sofiaParameters;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SofiaSearch, sofiaParameters)


auto runSearchSync(SofiaSearch const& search) -> SofiaResult
{
	auto childProcess = bp::child(boost::process::exe = sofiaPath, boost::process::args = search.sofiaParameters);
	auto result = SofiaResult{};
	if (childProcess.valid())
	{
		childProcess.wait();
		result.returnCode = childProcess.exit_code();
		result.finished = true;
	}

	return result;
}

static auto runSearch(SofiaSearch search) -> std::future<SofiaResult>
{
	return std::async(std::launch::async, runSearchSync, std::move(search));
}

enum class RequestState
{
	undefined,
	created,
	sofia_started,
	done
};

NLOHMANN_JSON_SERIALIZE_ENUM(RequestState,
							 {
								 { RequestState::undefined, nullptr },
								 { RequestState::created, "created" },
								 { RequestState::sofia_started, "sofia_started" },
								 { RequestState::done, "done" },
							 })

struct RequestResults
{
	SofiaResult sofiaResult;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RequestResults, sofiaResult)


class SofiaRequest
{
	public:
	SofiaRequest(std::string searchIdentifier, SofiaSearch sofiaSearch) : searchIdentifier(searchIdentifier), search(std::move(sofiaSearch))
		{
			if (!searchIdentifier.empty())
			{
				currentState = RequestState::created;
			}
		}

		auto process()
		{
			switch (currentState)
			{
			case RequestState::created:
				sofiaRun = runSearch(this->search);
				// currentProcess = bp::child(sofiaPath, search.sofiaParameters);
				
				currentState = RequestState::sofia_started;
				currentMessage = "SoFiA search started.";
				break;
			case RequestState::sofia_started:
				checkSearch();
				break;
			// case RequestState::done:
			default:
				break;
			}
		}

		auto getMessage() -> std::string_view
		{
			return currentMessage;
		}

		auto getResults() const -> RequestResults // copy
		{
			return results;
		}

		auto isValid() const -> bool
		{
			return currentState != RequestState::undefined;
		}

		auto isDone() const -> bool
		{
			return currentState == RequestState::done;
		}

		auto getSearchIdentifier() -> std::string_view
		{
			return searchIdentifier;
		}

	private:
		RequestState currentState{ RequestState::undefined };
		std::string searchIdentifier;

		SofiaSearch search;
		RequestResults results {};

		std::future<SofiaResult> sofiaRun;
		bp::child currentProcess;

		std::string currentMessage{ "Request is undefined." };

		auto checkSearch() -> void
		{
			using namespace std::chrono_literals;
			/*
			if (!currentProcess.valid() || currentProcess.running())
			{
				return;
			}

			currentProcess.wait();
			
			results.sofiaResult = { { currentProcess.exit_code() } };

			*/
			
			if (!sofiaRun.valid())
			{
				return;
			}
			const auto waitResult = sofiaRun.wait_for(0s);
			if (waitResult != std::future_status::ready)
			{
				return;
			}
			
			results.sofiaResult = sofiaRun.get();
			currentMessage = "SoFiA search finished.";
			currentState = RequestState::done;
		}
};

std::unique_ptr<SofiaRequest> currentRequest{ nullptr };
auto requestResults = std::unordered_map<std::string, RequestResults>();
std::mutex currentRequestMutex;

auto processCurrentRequest()-> void
{
	std::lock_guard lock(currentRequestMutex);
	if (!currentRequest)
	{
		return;
	}

	if (!currentRequest->isValid())
	{
		currentRequest.reset();
		std::cerr << "Invalid request processed was not valid\n";
	}

	currentRequest->process();

	if(!currentRequest->isDone())
	{
		return;
	}
	auto res = std::pair<std::string, RequestResults>{ currentRequest->getSearchIdentifier(), currentRequest->getResults() };
	requestResults.emplace(res);
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
		return EXIT_FAILURE;
	}
	std::cout << "Using " << sofiaPath << " as SoFiA executable\n";

	if (commonRootPath.empty())
	{
		std::cerr << "No common root path!\n";
		return EXIT_FAILURE;
	}
	std::cout << "Using " << commonRootPath << " as common root path\n";


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

	svr.Post("/start",
			 [](const httplib::Request& req, httplib::Response& res,
							  const httplib::ContentReader& content_reader)
			 {
				 processCurrentRequest();

			 	 std::lock_guard currRequestLock(currentRequestMutex);

				 // Ongoing request
				 if (currentRequest)
				 {
					 nlohmann::json retJ;
					 retJ["message"] = currentRequest->getMessage();
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
				if (jsonInput.empty() || !jsonInput.contains("search_identifier"))
				{
					nlohmann::json retJ;
					retJ["message"] = "Parameters empty or search_identifier not provided";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}


				std::string requestedSearchIdentifier = jsonInput["search_identifier"];

				// Identifier already used (Same Request)
				if (requestResults.find(requestedSearchIdentifier) != requestResults.end())
				{
					nlohmann::json retJ;
					retJ["message"] = "search_identifier already in use.";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				// Build new search
				SofiaSearch ss;
				if (jsonInput.contains("sofia_config_file"))
				{
					const auto fullPathString =
						(commonRootPath / std::filesystem::path(jsonInput["sofia_config_file"].get<std::string>()));
					ss.sofiaParameters.emplace_back(jsonInput["sofia_config_file"].get<std::string>());
				}

				for (auto& [key, value] : jsonInput["sofia_params"].items())
				{
					if (std::ranges::find(sofia_parameter_keys, key) !=
						sofia_parameter_keys.end())
					{
						// is path like
						if (std::ranges::find(sofia_path_parameter_keys, key) != sofia_path_parameter_keys.end())
						{
							auto inputStringForPath = value.get<std::string>();
							while (inputStringForPath.starts_with(".") || inputStringForPath.starts_with("/") ||
								   inputStringForPath.starts_with("\\"))
							{
								inputStringForPath.erase(0, 1);
							}

							const auto fullPathString =
								(commonRootPath / boost::process::filesystem::path(inputStringForPath)).string();



							ss.sofiaParameters.emplace_back(std::format(
								"{}={}", key.c_str(), fullPathString));
						}
						else
						{
							ss.sofiaParameters.emplace_back(std::format("{}={}", key.c_str(), value.get<std::string>()));
						}
					}
				}

				// Add new Request to currentRequest
				currentRequest = std::make_unique<SofiaRequest>(requestedSearchIdentifier, ss);
				currentRequest->process();

				res.set_content({}, "application/json");

			 });


	svr.Post("/result",
			 [](const httplib::Request& req, httplib::Response& res,
							const httplib::ContentReader& content_reader)
			 {
				 processCurrentRequest();
				 std::lock_guard currRequestLock(currentRequestMutex);

				

				std::string bodyString;
				 content_reader(
					 [&bodyString](const char* data, size_t data_length)
					 {
						 bodyString.append(data, data_length);
						 return true;
					 });

				 auto jsonInput = nlohmann::json::parse(bodyString);




				 // Input not valid
				 if (jsonInput.empty() || !jsonInput.contains("search_identifier"))
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "search_identifier not provided";

					 res.status = httplib::StatusCode::BadRequest_400;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }

				 std::string requestedSearchIdentifier = jsonInput["search_identifier"];

				 // requestedSearchIdentifier is Ongoing request
				 if (currentRequest && currentRequest->getSearchIdentifier() == requestedSearchIdentifier)
				 {
					 nlohmann::json retJ;
					 retJ["message"] = std::format("Request not finished yet");
					 res.status = httplib::StatusCode::ServiceUnavailable_503;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }

				 auto findit = requestResults.find(requestedSearchIdentifier);

				 // Identifier not found
				 if (findit == requestResults.end())
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "Request with given search_identifier not found.";

					 res.status = httplib::StatusCode::BadRequest_400;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }

				nlohmann::json retJ;
				 retJ["result"] = findit->second;
				 res.status = httplib::StatusCode::OK_200;
				 res.set_content(retJ.dump(), "application/json");

			 });

	svr.Get("/results", [](const httplib::Request&, httplib::Response& res)
			{ res.set_content(nlohmann::json(requestResults).dump(), "application/json");
			});

	std::cout << "Server is listening on port " << args::get(serverListeningPortArgument) << "\n";
	svr.listen("0.0.0.0", args::get(serverListeningPortArgument));
}
