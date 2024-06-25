#include <string>
#include <vector>

#include <httplib.h>
#include <map>

#include <nlohmann/json.hpp>

#include <filesystem>

#include <future>


#include <boost/process.hpp>

namespace bp = boost::process;

auto sofiaPath = boost::process::search_path("sofia");

const std::vector<std::string> sofia_return_code_messages = {
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
	
	if (sofiaPath.empty())
	{
		sofiaPath = boost::filesystem::path{ "D:/vcpkg/vcpkg.exe" };
	}

	auto params = std::vector<std::string>(argc);
	for (auto i = 0; i < argc; i++)
	{
		params[i] = argv[i];
	}

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
				for (auto& [key, value] : jsonInput["sofia_params"].items())
				{
					ss.sofiaParameters.emplace_back(
						std::format("{}={}", key.c_str(), value.get<std::string>()));
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
					 retJ["message"] = "search_identifier not provided";

					 res.status = httplib::StatusCode::BadRequest_400;
					 res.set_content(retJ.dump(), "application/json");
					 return;
				 }


				 std::string requestedSearchIdentifier = jsonInput["search_identifier"];
				 auto findit = requestResults.find(requestedSearchIdentifier);


				 // Identifier not found
				 if (findit == requestResults.end())
				 {
					 nlohmann::json retJ;
					 retJ["message"] = "search_identifier not found.";

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

	svr.listen("localhost", 8080);

}
