#include "SoFiASubregionFeature.h"

#include <future>

#include "RendererBase.h"

#include "httplib.h"

#include <nlohmann/json.hpp>


using namespace b3d::renderer;

namespace b3d::renderer 
{
	static void to_json(nlohmann::json& j, const SoFiaRequest& request)
	{
		j = nlohmann::json{ { "input.data", request.fitsInputPath } };
		j["input.region"] =
			std::format("{},{},{},{},{},{}", request.regionToSearch.lower.x, request.regionToSearch.upper.x,
						request.regionToSearch.lower.y, request.regionToSearch.upper.y, request.regionToSearch.lower.z,
						request.regionToSearch.upper.z);
	}
}


SoFiASubregionFeature::SoFiASubregionFeature(const std::string& name, b3d::renderer::RendererBase* renderer)
	: RenderFeature(name), renderer_(renderer)
{
}

void SoFiASubregionFeature::initialize(b3d::renderer::RenderingDataBuffer& sharedParameters)
{
	RenderFeature::initialize(sharedParameters);
	debugDrawlist_ = &renderer_->gizmoHelperBase();
}


auto SoFiASubregionFeature::gui() -> void
{
	using namespace std::chrono_literals;
	const auto volTransform = sharedParameters_->get<VolumeTransform>("volumeTransform");
	const auto transformedWorldMat = volTransform->worldMatTRS * volTransform->renormalizedScale * owl::AffineSpace3f::scale(volTransform->volumeVoxelBox.size());
	
	debugDrawlist_->drawBoundGizmo(boxTransform_, transformedWorldMat, {1,1,1});

	if (currentSearch.valid())
	{
		if (currentSearch.wait_for(0s) == std::future_status::ready)
		{
			auto la = currentSearch.get();
			lastSearch = std::make_unique<SoFiaSearch>(std::move(la));
		}
	}

	if (lastSearch && !lastSearch->result.message.empty())
	{
		ImGui::Text(lastSearch->result.message.c_str());
	}

	if (searchMutex_.try_lock())
	{
		if (ImGui::Button("Start Search"))
		{
			auto lu = prepareAndExecuteSearch(volTransform->volumeVoxelBox);
			ImGui::Button("Top");
		}
		searchMutex_.unlock();
	}
	else
	{
		if(ImGui::Button("Abort Search"))
		{
			stopSearch_ = true;
		}
	}



	auto scale = owl::vec3f{ length(boxTransform_.l.vx), length(boxTransform_.l.vx), length(boxTransform_.l.vx) };
	auto position = boxTransform_.p;

	auto lower = owl::vec3f{ -.5f, -.5f, -.5f };
	auto upper = owl::vec3f{ .5f, .5f, .5f };

	owl::vec3f lowerPos = xfmPoint(boxTransform_, lower) + owl::vec3f{ .5, .5, .5 };
	owl::vec3f upperPos = xfmPoint(boxTransform_, upper) + owl::vec3f{ .5, .5, .5 };

	owl::box3f regionBox = intersection(owl::box3f{ lowerPos, upperPos }, owl::box3f{{0,0,0},{1,1,1}});
	
	ImGui::BeginDisabled();
	ImGui::InputFloat3("Scale", &scale.x);
	ImGui::InputFloat3("Position", &position.x);

	ImGui::InputFloat3("LowerTransformed", &regionBox.lower.x);
	ImGui::InputFloat3("UpperTransformed", &regionBox.upper.x);

	ImGui::EndDisabled();
}

auto SoFiASubregionFeature::getParamsData() -> ParamsData
{
	return {  };
}

auto b3d::renderer::SoFiASubregionFeature::hasGui() const -> bool
{
	return true;
}

// Call this only if you own the mutex!
auto SoFiASubregionFeature::prepareAndExecuteSearch(owl::box3f volumeVoxelBox) -> SoFiaRequest
{
	std::filesystem::path relativePathToFileForNow = "n4565/n4565_lincube_big.fits";

	auto lower = owl::vec3f{ -.5f, -.5f, -.5f };
	auto upper = owl::vec3f{ .5f, .5f, .5f };

	owl::vec3f lowerPos = max(xfmPoint(boxTransform_, lower) + upper, owl::vec3f{ 0, 0, 0 });
	owl::vec3f upperPos = min(xfmPoint(boxTransform_, upper) + upper, owl::vec3f{ 1, 1, 1 });

	const auto lowerVoxel = lowerPos * volumeVoxelBox.size();
	const auto upperVoxel = upperPos * volumeVoxelBox.size();

	owl::box3i voxelBox = { { lround(lowerVoxel.x), lround(lowerVoxel.y), lround(lowerVoxel.z) },
							{ lround(upperVoxel.x), lround(upperVoxel.y), lround(upperVoxel.z) } };


	const auto request = SoFiaRequest{ .regionToSearch = voxelBox, .fitsInputPath = relativePathToFileForNow };

	stopSearch_ = false;
	currentSearch = std::async(&SoFiASubregionFeature::search, this, request, std::ref(stopSearch_));

	return request;

}

SoFiaSearch SoFiASubregionFeature::search(SoFiaRequest request, std::atomic_bool &stopToken)
{
	using namespace std::chrono_literals;
	const std::lock_guard<std::mutex> lock(searchMutex_);
	
	SoFiaSearch search = { .request = request };

	httplib::Client cli("127.0.0.1", 8080);

	nlohmann::json startRequest = request;
	auto startResponse = cli.Post("/start", startRequest.dump(), "application/json");
	if (startResponse.error() != httplib::Error::Success)
	{
		search.result.wasSuccess = false;
		search.result.message = "Error while requesting endpoint start";
		return search;
	}

	auto startResponseJson = nlohmann::json::parse(startResponse->body);
	if (startResponse->status != 200)
	{
		search.result.wasSuccess = false;
		search.result.message = startResponseJson["message"];
		return search;
	}

	search.searchHash = startResponseJson["search_hash"];


	auto resultRequest = nlohmann::json{ { "search_hash", search.searchHash } }.dump();

	bool stop = false;
	while (!stop && !stopToken)
	{
		auto resultResponse = cli.Post("/result", resultRequest, "application/json");
		if (resultResponse.error() != httplib::Error::Success)
		{
			search.result.wasSuccess = false;
			search.result.message = "Error while requesting endpoint result";
			return search;
		}

		auto resultResponseJson = nlohmann::json::parse(resultResponse->body);
		switch (resultResponse->status)
		{
			case 503:

				std::this_thread::sleep_for(200ms);
				continue;
			case 200:
				search.result.wasSuccess = true;
				search.result.message = resultResponseJson["message"];
				break;
			case 400:
				search.result.message = resultResponseJson["message"];
				search.result.wasSuccess = false;
				break;
			default:
				search.result.wasSuccess = false;
				search.result.message = "Unknown error";
				break;
		}
		stop = true;
	}
	if (stopToken)
	{
		search.result.message += "Aborted by User.";
	}

	return search;
}

SoFiASubregionFeature::~SoFiASubregionFeature()
{
	stopSearch_ = true;
	if (currentSearch.valid())
	{
		currentSearch.wait();
	}
}
