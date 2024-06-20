#include <string>
#include <vector>

#include <httplib.h>
#include <map>

#include <nlohmann/json.hpp>

#include <filesystem>


#include <boost/process.hpp>

namespace bp = boost::process;

auto sofiaPath = bp::search_path("vcpk");


const std::vector<std::string> sofia_return_code_messages = {
	"The pipeline successfully completed without any error.",
	"An unclassified failure occurred.",
	"A NULL pointer was encountered.",
	"A memory allocation error occurred. This could indicate that the data cube is too large for the amount of memory "
	"available on the machine.",
	"An array index was found to be out of range.",
	"An error occurred while trying to read or write a file or check if a directory or file is accessible.",
	"The overflow of an integer value occurred.",
	"The pipeline had to be aborted due to invalid user input. This could, e.g., be due to an invalid parameter "
	"setting or the wrong input file being provided.",
	"No specific error occurred, but sources were not detected either."
};

struct SofiaSearch
{
	std::vector<std::string> sofiaParameters;
	std::string searchIdentifier;

	bool started = false;
	bool finished = false;
	bp::child process;
	

	auto startSearch(const boost::filesystem::path pathToExecutable) -> void
	{
		if (started)
		{
			return;
		}

		started = true;
		process = bp::child(boost::process::exe = pathToExecutable, boost::process::args = sofiaParameters);
	}

	auto isDone() -> bool
	{
		if (finished)
		{
			return true;
		}

		if (!started || process.running())
		{
			return false;
		}
		
		process.wait();
		finished = true;
		return true;
	}

	auto getMessage() -> std::string
	{
		const auto retCode = getCode();
		if (0 <= retCode && retCode < sofia_return_code_messages.size())
		{
			return sofia_return_code_messages[retCode];
		}
		// Not correct if return code is unknown (bigger than array)
		return "Running";
	}

	auto getCode() -> int
	{
		if (isDone())
		{
			return process.exit_code();
		}
		return -1;
	}
};

std::unique_ptr<SofiaSearch> currentSearch{ nullptr };


auto main(const int argc, char** argv) -> int
{
	auto allSearches = std::unordered_map<std::string, SofiaSearch>();
	
	if (sofiaPath.empty())
	{
		sofiaPath = boost::filesystem::path{ "D:/vcpkg/vcpkg.exe" };
	}

	bp::child process;

	std::cout << process.valid() << std::endl;
	
	auto childProcess = bp::child(boost::process::exe = sofiaPath, boost::process::args = { "main.cpp" });

	while (childProcess.running())
	{
		
	}

	childProcess.wait();

	auto params = std::vector<std::string>(argc);
	for (auto i = 0; i < argc; i++)
	{
		params[i] = argv[i];
	}

	httplib::Server svr;

	svr.Post("/start",
			 [&allSearches](const httplib::Request& req, httplib::Response& res,
							  const httplib::ContentReader& content_reader)
			 {
				 if (currentSearch)
				 {
					 if(!currentSearch->isDone())
					 {
						 nlohmann::json retJ;
						 retJ["message"] = "Ongoing search";

						 res.status = httplib::StatusCode::ServiceUnavailable_503;
						 res.set_content(retJ.dump(), "application/json");
					 }
					 else
					 {
						 allSearches[currentSearch->searchIdentifier] = std::move(*currentSearch);
					 }
				 }

				// json -> sofia input format
				 std::string body;
				 content_reader(
					 [&body](const char* data, size_t data_length)
					 {
						 body.append(data, data_length);
						 return true;
					 });

			 	
			 	auto jsonInput = nlohmann::json::parse(body);

				if (jsonInput.empty() || !jsonInput.contains("search_identifier"))
				{
					nlohmann::json retJ;
					retJ["message"] = "Parameters empty or search_identifier not provided";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}
				std::string requestedSearchIdentifier = jsonInput["search_identifier"];
				if (currentSearch && currentSearch->searchIdentifier == requestedSearchIdentifier ||
					allSearches.find(requestedSearchIdentifier) != allSearches.end())
				{
					nlohmann::json retJ;
					retJ["message"] = "search_identifier already in use.";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				currentSearch = std::make_unique<SofiaSearch>();
				for (auto& [key, value] : jsonInput["sofia_params"].items())
				{
					currentSearch->sofiaParameters.emplace_back(
						std::format("{}={}", key.c_str(), value.get<std::string>()));
				}

				currentSearch->searchIdentifier = requestedSearchIdentifier;
				currentSearch->startSearch(sofiaPath);

				auto jsonOutput = nlohmann::json();
				jsonOutput["stringified_input"] = currentSearch->sofiaParameters;

			 	res.set_content(jsonOutput.dump(), "application/json");
			 });


	svr.Post("/result",
			 [&allSearches](const httplib::Request& req, httplib::Response& res,
							const httplib::ContentReader& content_reader)
			 {	
				if (currentSearch)
				{
					allSearches[currentSearch->searchIdentifier] = std::move(*currentSearch);
				}

				std::string body;
				content_reader(
					[&body](const char* data, size_t data_length)
					{
						body.append(data, data_length);
						return true;
					});

				auto jsonInput = nlohmann::json::parse(body);

				if (jsonInput.empty() || !jsonInput.contains("search_identifier"))
				{
					nlohmann::json retJ;
					retJ["message"] = "search_identifier not provided";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				std::string requestedSearchIdentifier = jsonInput["search_identifier"];
				if (currentSearch && currentSearch->searchIdentifier == requestedSearchIdentifier)
				{
					nlohmann::json retJ;
					retJ["message"] = "Requested search is running.";

					res.status = httplib::StatusCode::ServiceUnavailable_503;
					res.set_content(retJ.dump(), "application/json");
					return;
				}

				auto findit = allSearches.find(requestedSearchIdentifier);

				if (findit == allSearches.end())
				{
					nlohmann::json retJ;
					retJ["message"] = "Search with given identifier not found.";

					res.status = httplib::StatusCode::BadRequest_400;
					res.set_content(retJ.dump(), "application/json");
				}


				nlohmann::json retJ;
				retJ["message"] = findit->second.getMessage();
				retJ["error_code"] = findit->second.getCode();
				res.status = httplib::StatusCode::OK_200;
				res.set_content(retJ.dump(), "application/json");
			 });

	svr.listen("localhost", 8080);

}
